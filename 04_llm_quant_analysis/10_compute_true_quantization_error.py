"""
10_compute_true_quantization_error.py

목적: "진짜" 양자화 오차(BF16 원본 모델과 GPTQ 양자화 모델의 같은 레이어·같은 위치
출력 텐서 사이의 차이)를 계산한다.

기존 layer_statistics.json(02_cuda_aligned/05_controlled_inference_eval.py가 생성)은
단일 모델 자신의 활성화 L2 norm(attn_global_l2_norm, mlp_global_l2_norm)일 뿐, 두 모델
사이의 차이를 계산한 적이 없다는 사실이 사전 분석에서 확인되었다. 이 스크립트는 그
차이(delta)를 실제로 계산하는 별도의 새 스크립트이며:

  - 기존 파일(05_controlled_inference_eval.py, layer_statistics.json, 모든 분석
    노트북)은 절대 수정하거나 덮어쓰지 않는다.
  - 계산 결과는 각 (모델, bit level, 프롬프트) 폴더 안에 새 파일
    "layer_delta_statistics.json"으로만 저장한다.
  - 이미 디스크에 저장되어 있는 layer_{i}_attn_output.pt / layer_{i}_mlp_output.pt
    텐서만 읽어서 사후 계산하며, 모델이나 GPU를 다시 로드하지 않는다.

출력: 계산 결과 저장 경로 목록 + 자체 검증(sanity check) 결과만 콘솔에 출력한다.
그래프/시각화/해석은 이 스크립트의 범위가 아니다.
"""

import os
import sys
import json
import math

import numpy as np
import pandas as pd
import torch

# Windows 콘솔의 기본 코드페이지(cp949 등)는 이 스크립트가 출력하는 일부 문자
# (예: em dash "—", 화살표 "→")를 인코딩하지 못해 UnicodeEncodeError로 죽는다.
# 표준출력/표준에러를 UTF-8로 강제해 어떤 콘솔에서 실행해도 죽지 않게 한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# 0. 설정
# ==============================================================================
BASE_DIR = r"D:\slm-gptq-collapse-analysis\02_cuda_aligned\Experiment_Data_v2"

MODELS = {
    "Llama-3.2-1B-Instruct": 16,
    "Qwen2.5-1.5B-Instruct": 28,
}

REFERENCE_BIT = "Original_BF16"
COMPARISON_BITS = ["GPTQ_8bit", "GPTQ_4bit", "GPTQ_3bit", "GPTQ_2bit"]  # 순서 = 정밀도 감소 순
PROMPTS = ["Prompt_01", "Prompt_02", "Prompt_03", "Prompt_04"]
BLOCK_TYPES = ["attn", "mlp"]

RELATIVE_ERROR_8BIT_THRESHOLD = 0.05  # 사용자가 지정한 8-bit 기준치


def experiment_dir(model_name: str, bit_level: str) -> str:
    return os.path.join(BASE_DIR, f"{model_name}_{bit_level}")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 1단계: 입력 시퀀스 동일성 확인
# ==============================================================================
def check_input_sequences() -> list:
    """
    02_cuda_aligned/05_controlled_inference_eval.py의 evaluate_model() 구조상,
    GPTQ 실험 폴더는 자기 자신의 input_ids 텐서를 저장하지 않는다. 대신
    reference_ids_path = SAVE_BASE_DIR/{model}_Original_BF16/{prompt}/reference_input_ids.pt
    를 그대로 torch.load()해서 재사용한다(해당 파일 559~584줄). 즉 GPTQ 폴더 쪽에는
    직접 diff를 뜰 별도의 input_ids 파일이 애초에 존재하지 않는다.

    그래서 이 함수는:
      (a) reference_input_ids.pt가 실제로 Original_BF16 폴더에만 있고 GPTQ 폴더에는
          없다는 것(=파일을 공유해서 쓰는 구조라는 것)을 먼저 사실로 확인하고,
      (b) 각 실험의 tensor_metadata.json에 기록된 input_sequence_length /
          full_sequence_length가 Original_BF16과 동일한지 비교한다.
          (05_controlled_inference_eval.py 568~582줄은 이 두 값이 실제로 다르면
          RuntimeError를 던지고 데이터 생성 자체를 중단시키므로, 데이터가 존재한다는
          것 자체가 이미 하나의 검증이지만 여기서는 그 결과를 직접 다시 확인한다.)

    길이가 다르면 그 자체가 "입력이 달랐다"는 확정적 증거이므로 즉시 보고한다.
    길이가 같다고 해서 토큰 값까지 100% 동일하다고 수학적으로 보장하는 것은 아니다
    (그 보장은 위에서 언급한 생성 시점의 assert가 담당한다) — 이 한계는 아래 출력에
    그대로 명시한다.
    """
    print("=" * 78)
    print("[1단계] 입력 시퀀스 동일성 확인")
    print("=" * 78)

    rows = []
    mismatches = []

    for model_name in MODELS:
        for prompt_id in PROMPTS:
            ref_meta = load_json(os.path.join(experiment_dir(model_name, REFERENCE_BIT), prompt_id, "tensor_metadata.json"))
            ref_ids_path = os.path.join(experiment_dir(model_name, REFERENCE_BIT), prompt_id, "reference_input_ids.pt")
            ref_ids_exists = os.path.exists(ref_ids_path)

            if ref_meta is None:
                mismatches.append((model_name, REFERENCE_BIT, prompt_id, "reference tensor_metadata.json 없음"))
                continue

            ref_lens = (ref_meta.get("input_sequence_length"), ref_meta.get("full_sequence_length"))

            for bit_level in COMPARISON_BITS:
                cmp_meta = load_json(os.path.join(experiment_dir(model_name, bit_level), prompt_id, "tensor_metadata.json"))
                own_ids_path = os.path.join(experiment_dir(model_name, bit_level), prompt_id, "reference_input_ids.pt")
                own_ids_exists = os.path.exists(own_ids_path)

                if cmp_meta is None:
                    mismatches.append((model_name, bit_level, prompt_id, "tensor_metadata.json 없음"))
                    continue

                cmp_lens = (cmp_meta.get("input_sequence_length"), cmp_meta.get("full_sequence_length"))
                same = ref_lens == cmp_lens

                rows.append({
                    "model": model_name,
                    "bit_level": bit_level,
                    "prompt": prompt_id,
                    "ref_input_len": ref_lens[0],
                    "ref_full_len": ref_lens[1],
                    "cmp_input_len": cmp_lens[0],
                    "cmp_full_len": cmp_lens[1],
                    "same_length": same,
                    "bf16_has_own_reference_input_ids.pt": ref_ids_exists,
                    "gptq_has_own_reference_input_ids.pt": own_ids_exists,
                })

                if not same:
                    mismatches.append((model_name, bit_level, prompt_id,
                                        f"길이 불일치: BF16={ref_lens} vs {bit_level}={cmp_lens}"))
                if own_ids_exists:
                    # 코드 상 절대 만들어지지 않아야 하는 파일. 있으면 구조 가정이
                    # 깨진 것이므로 별도로 강하게 보고한다.
                    mismatches.append((model_name, bit_level, prompt_id,
                                        "예상과 달리 GPTQ 폴더가 자체 reference_input_ids.pt를 갖고 있음 "
                                        "— BF16 것과 별도로 직접 diff 확인이 필요함"))

    df = pd.DataFrame(rows)
    if not df.empty:
        print(df.to_string(index=False))
    print()

    if mismatches:
        print(f"[결과] 문제 {len(mismatches)}건 발견:")
        for m in mismatches:
            print("  -", m)
    else:
        print("[결과] 모든 (model, bit_level, prompt) 조합에서 tensor_metadata.json의 "
              "input_sequence_length / full_sequence_length가 Original_BF16과 동일했습니다.")
        print("[한계] GPTQ 실험 폴더는 자체 input_ids 텐서 파일이 없고 Original_BF16의 "
              "reference_input_ids.pt를 그대로 재사용하는 구조라서, 파일 자체를 바이트 단위로 "
              "diff하는 것은 애초에 불가능합니다. 위 검사는 '길이가 다르다'는 오류는 확실히 "
              "잡아내지만, '길이는 같은데 토큰 값 자체가 달랐다'는 경우까지 이 스크립트 선에서 "
              "재검증하지는 못합니다(그 보장은 데이터 생성 스크립트의 런타임 assert가 담당).")
    print()

    return mismatches


# ==============================================================================
# 2~3단계: 델타(진짜 양자화 오차) 계산 및 저장
# ==============================================================================
def load_module_tensor(model_name, bit_level, prompt_id, layer_idx, block_type):
    path = os.path.join(experiment_dir(model_name, bit_level), prompt_id, "tensors",
                         f"layer_{layer_idx}_{block_type}_output.pt")
    if not os.path.exists(path):
        return None, path
    tensor = torch.load(path, map_location="cpu")
    return tensor, path


def compute_delta_for_experiment(model_name, bit_level, prompt_id, num_layers):
    layer_records = []
    problems = []

    for layer_idx in range(num_layers):
        for block_type in BLOCK_TYPES:
            bf16_t, bf16_path = load_module_tensor(model_name, REFERENCE_BIT, prompt_id, layer_idx, block_type)
            gptq_t, gptq_path = load_module_tensor(model_name, bit_level, prompt_id, layer_idx, block_type)

            if bf16_t is None or gptq_t is None:
                problems.append(
                    f"{model_name}/{bit_level}/{prompt_id} layer={layer_idx} {block_type}: "
                    f"텐서 파일 없음 (bf16_exists={bf16_t is not None}, gptq_exists={gptq_t is not None})"
                )
                continue

            if tuple(bf16_t.shape) != tuple(gptq_t.shape):
                problems.append(
                    f"{model_name}/{bit_level}/{prompt_id} layer={layer_idx} {block_type}: "
                    f"shape 불일치 bf16={tuple(bf16_t.shape)} vs gptq={tuple(gptq_t.shape)} — 계산 건너뜀"
                )
                continue

            bf16_f = bf16_t.float()
            gptq_f = gptq_t.float()
            delta = bf16_f - gptq_f

            abs_l2 = delta.norm().item()
            bf16_norm = bf16_f.norm().item()
            gptq_norm = gptq_f.norm().item()
            relative_error = (abs_l2 / bf16_norm) if bf16_norm != 0 else float("nan")
            norm_ratio = (gptq_norm / bf16_norm) if bf16_norm != 0 else float("nan")

            layer_records.append({
                "model": model_name,
                "reference_bit_level": REFERENCE_BIT,
                "comparison_bit_level": bit_level,
                "prompt": prompt_id,
                "layer": layer_idx,
                "block_type": block_type,
                "bf16_path": bf16_path,
                "gptq_path": gptq_path,
                "abs_l2": abs_l2,
                "bf16_norm": bf16_norm,
                "gptq_norm": gptq_norm,
                "relative_error": relative_error,
                "norm_ratio": norm_ratio,
            })

    return layer_records, problems


def save_layer_delta_statistics(model_name, bit_level, prompt_id, layer_records):
    out_dir = os.path.join(experiment_dir(model_name, bit_level), prompt_id)
    out_path = os.path.join(out_dir, "layer_delta_statistics.json")

    existing_stats_path = os.path.join(out_dir, "layer_statistics.json")
    assert os.path.exists(existing_stats_path), \
        f"예상과 다른 폴더 구조: {existing_stats_path} 가 없음 — 저장 위치를 다시 확인할 것"

    payload = {
        "reference_experiment": f"{model_name}_{REFERENCE_BIT}",
        "comparison_experiment": f"{model_name}_{bit_level}",
        "prompt_id": prompt_id,
        "note": (
            "True cross-model quantization error (NOT single-model activation norm). "
            "For each decoder layer / block_type in {attn, mlp}: "
            "delta = bf16_tensor.float() - gptq_tensor.float(); "
            "abs_l2 = ||delta||_2; bf16_norm = ||bf16||_2; gptq_norm = ||gptq||_2; "
            "relative_error = abs_l2 / bf16_norm; norm_ratio = gptq_norm / bf16_norm. "
            "Generated by 10_compute_true_quantization_error.py. "
            "This file is separate from, and does not overwrite, layer_statistics.json "
            "(which stores each single model's own activation L2 norm, not a delta)."
        ),
        "layers": layer_records,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path


def run_all_delta_computations():
    print("=" * 78)
    print("[2~3단계] BF16 vs GPTQ 델타 계산 및 layer_delta_statistics.json 저장")
    print("=" * 78)

    all_records = []
    all_problems = []
    saved_paths = []

    for model_name, num_layers in MODELS.items():
        for bit_level in COMPARISON_BITS:
            for prompt_id in PROMPTS:
                layer_records, problems = compute_delta_for_experiment(
                    model_name, bit_level, prompt_id, num_layers
                )
                all_records.extend(layer_records)
                all_problems.extend(problems)

                if layer_records:
                    out_path = save_layer_delta_statistics(model_name, bit_level, prompt_id, layer_records)
                    saved_paths.append(out_path)
                    print(f"  저장 완료: {out_path}  ({len(layer_records)}개 레이어/블록 레코드)")
                else:
                    print(f"  [건너뜀] {model_name}/{bit_level}/{prompt_id}: 계산된 레코드 없음")

    print()
    print(f"총 저장된 layer_delta_statistics.json 파일 수: {len(saved_paths)}")
    if all_problems:
        print(f"계산 중 문제 {len(all_problems)}건:")
        for p in all_problems:
            print("  -", p)
    else:
        print("텐서 파일 누락/shape 불일치 문제 없음.")
    print()

    return pd.DataFrame(all_records), all_problems


# ==============================================================================
# 4단계: 자체 검증
# ==============================================================================
def self_check_a_no_self_comparison(df: pd.DataFrame):
    print("-" * 78)
    print("[자체검증 A] BF16을 자기 자신과 비교한 레코드가 있는가?")
    print("-" * 78)

    if df.empty:
        print("  레코드 없음 — 검증 대상 없음")
        print()
        return

    same_bit = df[df["reference_bit_level"] == df["comparison_bit_level"]]
    same_path = df[df["bf16_path"] == df["gptq_path"]]
    not_from_reference = df[df["reference_bit_level"] != REFERENCE_BIT]
    comparison_is_reference = df[df["comparison_bit_level"] == REFERENCE_BIT]

    n_problems = len(same_bit) + len(same_path) + len(not_from_reference) + len(comparison_is_reference)

    if n_problems == 0:
        print(f"  PASS: 전체 {len(df)}개 레코드 모두 reference_bit_level='{REFERENCE_BIT}' "
              f"!= comparison_bit_level 이고, bf16_path != gptq_path 였습니다.")
    else:
        print(f"  FAIL: 자기 자신과 비교된 것으로 보이는 레코드 {n_problems}건 발견")
        if len(same_bit) > 0:
            print(same_bit.to_string(index=False))
        if len(same_path) > 0:
            print(same_path.to_string(index=False))
    print()


def self_check_b_8bit_small_error(df: pd.DataFrame):
    print("-" * 78)
    print(f"[자체검증 B] GPTQ_8bit의 relative_error가 대부분 {RELATIVE_ERROR_8BIT_THRESHOLD} 미만인가?")
    print("-" * 78)

    df_8bit = df[df["comparison_bit_level"] == "GPTQ_8bit"]
    if df_8bit.empty:
        print("  GPTQ_8bit 레코드 없음")
        print()
        return

    total = len(df_8bit)
    n_small = (df_8bit["relative_error"] < RELATIVE_ERROR_8BIT_THRESHOLD).sum()
    frac = n_small / total

    print(f"  전체 GPTQ_8bit 레코드: {total}개")
    print(f"  relative_error < {RELATIVE_ERROR_8BIT_THRESHOLD}: {n_small}개 ({frac:.1%})")

    breakdown = (
        df_8bit.assign(is_small=df_8bit["relative_error"] < RELATIVE_ERROR_8BIT_THRESHOLD)
        .groupby(["model", "prompt", "block_type"])["is_small"]
        .agg(n_total="count", n_small="sum")
    )
    breakdown["frac_small"] = breakdown["n_small"] / breakdown["n_total"]
    print(breakdown.to_string())

    if frac < 0.9:
        print(f"  [경고] GPTQ_8bit인데도 relative_error < {RELATIVE_ERROR_8BIT_THRESHOLD}인 비율이 "
              f"{frac:.1%}로 낮습니다.")
    else:
        print(f"  [확인] 기준 통과 (임계값: 90% 이상 레이어가 {RELATIVE_ERROR_8BIT_THRESHOLD} 미만)")
    print()


def self_check_c_monotonic_by_bit(df: pd.DataFrame):
    print("-" * 78)
    print("[자체검증 C] bit level이 낮아질수록(8→4→3→2) relative_error가 대체로 증가하는가?")
    print("-" * 78)

    if df.empty:
        print("  레코드 없음")
        print()
        return

    bit_order = COMPARISON_BITS  # ["GPTQ_8bit", "GPTQ_4bit", "GPTQ_3bit", "GPTQ_2bit"]

    pivot = df.pivot_table(
        index=["model", "prompt", "block_type", "layer"],
        columns="comparison_bit_level",
        values="relative_error",
    )
    pivot = pivot.reindex(columns=bit_order)

    complete = pivot.dropna()

    def is_monotonic_nondecreasing(row):
        values = row.values
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    n_total = len(complete)
    n_monotonic = complete.apply(is_monotonic_nondecreasing, axis=1).sum() if n_total > 0 else 0
    n_endpoints_up = (complete[bit_order[-1]] >= complete[bit_order[0]]).sum() if n_total > 0 else 0

    print(f"  (model, prompt, block_type, layer) 조합 총 {n_total}개 중:")
    if n_total > 0:
        print(f"    8→4→3→2 전 구간에서 non-decreasing(단조 비감소)인 조합: "
              f"{n_monotonic}개 ({n_monotonic / n_total:.1%})")
        print(f"    양 끝(GPTQ_8bit → GPTQ_2bit)만 비교했을 때 증가한 조합: "
              f"{n_endpoints_up}개 ({n_endpoints_up / n_total:.1%})")
    else:
        print("    비교 가능한 완전한(4개 bit level 모두 존재) 조합 없음")

    print()
    print("  [결과 테이블] (model, block_type, prompt) 별 레이어 평균 relative_error"
          " — 프롬프트 간 평균 내지 않음, 프롬프트별로 각각 표시:")
    summary_table = (
        df.groupby(["model", "block_type", "prompt", "comparison_bit_level"])["relative_error"]
        .mean()
        .unstack("comparison_bit_level")
        .reindex(columns=bit_order)
    )
    print(summary_table.to_string())
    print()


def self_check_d_nan_inf(df: pd.DataFrame):
    print("-" * 78)
    print("[자체검증 D] 계산 결과에 NaN 또는 Inf가 있는가?")
    print("-" * 78)

    if df.empty:
        print("  레코드 없음")
        print()
        return

    numeric_cols = ["abs_l2", "bf16_norm", "gptq_norm", "relative_error", "norm_ratio"]
    problem_rows = []
    for col in numeric_cols:
        values = df[col].to_numpy(dtype=float)
        bad_mask = ~np.isfinite(values)  # NaN과 Inf/-Inf 모두 잡음
        n_bad = int(bad_mask.sum())
        print(f"  {col}: NaN/Inf {n_bad}개 / 전체 {len(df)}개")
        if n_bad > 0:
            bad_df = df.loc[bad_mask, ["model", "comparison_bit_level", "prompt", "layer", "block_type", col]]
            problem_rows.append(bad_df)

    if problem_rows:
        print()
        print("  문제 레코드 상세:")
        print(pd.concat(problem_rows).drop_duplicates().to_string(index=False))
    else:
        print("  [확인] NaN/Inf 없음")
    print()


def run_self_checks(df: pd.DataFrame):
    print("=" * 78)
    print("[4단계] 자체 검증")
    print("=" * 78)
    self_check_a_no_self_comparison(df)
    self_check_b_8bit_small_error(df)
    self_check_c_monotonic_by_bit(df)
    self_check_d_nan_inf(df)


# ==============================================================================
# 5단계: 프롬프트별 개별 결과 테이블 (평균 내지 않음)
# ==============================================================================
def print_per_prompt_result_table(df: pd.DataFrame):
    print("=" * 78)
    print("[5단계] Prompt_01~04 개별 결과 테이블 (프롬프트 간 평균 없음)")
    print("=" * 78)

    if df.empty:
        print("  레코드 없음")
        return

    table = df[[
        "model", "comparison_bit_level", "prompt", "layer", "block_type",
        "abs_l2", "bf16_norm", "gptq_norm", "relative_error", "norm_ratio",
    ]].sort_values(["model", "comparison_bit_level", "prompt", "layer", "block_type"])

    print(table.to_string(index=False))
    print()
    print(f"총 레코드 수: {len(table)}")


# ==============================================================================
# 실행
# ==============================================================================
if __name__ == "__main__":
    input_mismatches = check_input_sequences()

    df_all, problems = run_all_delta_computations()

    run_self_checks(df_all)

    print_per_prompt_result_table(df_all)
