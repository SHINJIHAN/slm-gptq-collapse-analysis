"""
11_dimension_token_concentration_analysis.py

목적: BF16-GPTQ 델타(진짜 양자화 오차)의 에너지(제곱합)가
  (1) 소수의 hidden dimension에 몰리는가 — Timkey et al. 2021, "rogue dimension"
  (2) 소수의 토큰 위치(특히 시퀀스 시작 부분)에 몰리는가 — Sun et al. 2024, "massive activation"
두 축을 모델 × bit level × 프롬프트 × 레이어 × block_type(attn/mlp)의 모든 조합에 대해
계산한다.

핵심 연산 (사용자가 제시한 식 그대로):
    delta = (t_bf16 - t_gptq).float().squeeze(0)   # [seq, dim]
    sq = delta ** 2
    dim_energy   = sq.sum(0)   # [dim]  -> topk(5)  : Timkey(rogue dimension) 검증
    token_energy = sq.sum(1)   # [seq]  -> topk(5)  : Sun(massive activation) 검증

기존 파일(05_controlled_inference_eval.py, layer_statistics.json,
10_compute_true_quantization_error.py, layer_delta_statistics.json, 모든 분석 노트북)은
전혀 건드리지 않는다. 이 스크립트는 새로 읽기만 하고, 새 출력 파일
"dimension_token_concentration_full.csv"만 만든다.

이 스크립트는 "얼마나 몰렸는지"를 판단(PASS/FAIL, 해석)하지 않는다. 단, top5가
"우연히 균등분포였다면 원래 차지했을 비중"(5/전체 개수)과 실측 비중을 나란히 보여줘서
읽는 사람이 직접 판단할 수 있는 기계적 숫자(concentration_x = 실측/균등기대)만 추가한다.
그래프/시각화/요약 없이, 계산된 모든 (model, bit, prompt, layer, block) 조합을 하나도
빠짐없이 표로 출력한다.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch

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
COMPARISON_BITS = ["GPTQ_8bit", "GPTQ_4bit", "GPTQ_3bit", "GPTQ_2bit"]
PROMPTS = ["Prompt_01", "Prompt_02", "Prompt_03", "Prompt_04"]
BLOCK_TYPES = ["attn", "mlp"]

TOPK = 5

OUTPUT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "dimension_token_concentration_full.csv")


def experiment_dir(model_name: str, bit_level: str) -> str:
    return os.path.join(BASE_DIR, f"{model_name}_{bit_level}")


def load_module_tensor(model_name, bit_level, prompt_id, layer_idx, block_type):
    path = os.path.join(experiment_dir(model_name, bit_level), prompt_id, "tensors",
                         f"layer_{layer_idx}_{block_type}_output.pt")
    if not os.path.exists(path):
        return None, path
    tensor = torch.load(path, map_location="cpu")
    return tensor, path


# ==============================================================================
# 핵심 연산: 사용자가 제시한 식 그대로
# ==============================================================================
def analyze_dim_token_concentration(bf16_t: torch.Tensor, gptq_t: torch.Tensor, topk: int = TOPK) -> dict:
    delta = (bf16_t.float() - gptq_t.float()).squeeze(0)  # [seq, dim]
    sq = delta ** 2                                        # [seq, dim]

    seq_len, hidden_dim = sq.shape

    dim_energy = sq.sum(0)     # [dim]  Timkey(rogue dimension) 축
    token_energy = sq.sum(1)   # [seq]  Sun(massive activation) 축
    total_energy = sq.sum().item()

    k_dim = min(topk, hidden_dim)
    k_tok = min(topk, seq_len)

    dim_top_vals, dim_top_idx = dim_energy.topk(k_dim)
    tok_top_vals, tok_top_idx = token_energy.topk(k_tok)

    dim_top_idx_list = dim_top_idx.tolist()
    tok_top_idx_list = tok_top_idx.tolist()

    dim_top_share = (dim_top_vals.sum().item() / total_energy) if total_energy > 0 else float("nan")
    tok_top_share = (tok_top_vals.sum().item() / total_energy) if total_energy > 0 else float("nan")

    dim_uniform_share = k_dim / hidden_dim
    tok_uniform_share = k_tok / seq_len

    return {
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "total_sq_energy": total_energy,

        # Timkey(1-1): 차원별 상위 기여 — 소수 dim에 몰리는가
        "dim_top5_idx": dim_top_idx_list,
        "dim_top5_val": [round(v, 6) for v in dim_top_vals.tolist()],
        "dim_top5_share": dim_top_share,
        "dim_uniform_expected_share": dim_uniform_share,
        "dim_concentration_x": (dim_top_share / dim_uniform_share) if dim_uniform_share > 0 else float("nan"),

        # Sun(1-4): 토큰별 상위 기여 — 시작 토큰에 몰리는가
        "token_top5_idx": tok_top_idx_list,
        "token_top5_val": [round(v, 6) for v in tok_top_vals.tolist()],
        "token_top5_share": tok_top_share,
        "token_uniform_expected_share": tok_uniform_share,
        "token_concentration_x": (tok_top_share / tok_uniform_share) if tok_uniform_share > 0 else float("nan"),
        "token_top5_min_position": min(tok_top_idx_list),
        "token_top5_contains_pos0": 0 in tok_top_idx_list,
    }


def list_to_str(values, ndigits=None):
    if ndigits is None:
        return ",".join(str(v) for v in values)
    return ",".join(f"{v:.{ndigits}f}" for v in values)


# ==============================================================================
# 전체 조합 순회
# ==============================================================================
def run_all_combinations():
    rows = []
    problems = []

    for model_name, num_layers in MODELS.items():
        for bit_level in COMPARISON_BITS:
            for prompt_id in PROMPTS:
                for layer_idx in range(num_layers):
                    for block_type in BLOCK_TYPES:
                        bf16_t, bf16_path = load_module_tensor(
                            model_name, REFERENCE_BIT, prompt_id, layer_idx, block_type
                        )
                        gptq_t, gptq_path = load_module_tensor(
                            model_name, bit_level, prompt_id, layer_idx, block_type
                        )

                        if bf16_t is None or gptq_t is None:
                            problems.append(
                                f"{model_name}/{bit_level}/{prompt_id} layer={layer_idx} {block_type}: "
                                f"텐서 파일 없음 (bf16_exists={bf16_t is not None}, gptq_exists={gptq_t is not None})"
                            )
                            continue

                        if tuple(bf16_t.shape) != tuple(gptq_t.shape):
                            problems.append(
                                f"{model_name}/{bit_level}/{prompt_id} layer={layer_idx} {block_type}: "
                                f"shape 불일치 bf16={tuple(bf16_t.shape)} vs gptq={tuple(gptq_t.shape)} — 건너뜀"
                            )
                            continue

                        result = analyze_dim_token_concentration(bf16_t, gptq_t, topk=TOPK)

                        rows.append({
                            "model": model_name,
                            "comparison_bit_level": bit_level,
                            "prompt": prompt_id,
                            "layer": layer_idx,
                            "block_type": block_type,
                            "seq_len": result["seq_len"],
                            "hidden_dim": result["hidden_dim"],
                            "total_sq_energy": result["total_sq_energy"],

                            "dim_top5_idx": list_to_str(result["dim_top5_idx"]),
                            "dim_top5_val": list_to_str(result["dim_top5_val"]),
                            "dim_top5_share": result["dim_top5_share"],
                            "dim_uniform_expected_share": result["dim_uniform_expected_share"],
                            "dim_concentration_x": result["dim_concentration_x"],

                            "token_top5_idx": list_to_str(result["token_top5_idx"]),
                            "token_top5_val": list_to_str(result["token_top5_val"]),
                            "token_top5_share": result["token_top5_share"],
                            "token_uniform_expected_share": result["token_uniform_expected_share"],
                            "token_concentration_x": result["token_concentration_x"],
                            "token_top5_min_position": result["token_top5_min_position"],
                            "token_top5_contains_pos0": result["token_top5_contains_pos0"],
                        })

    return pd.DataFrame(rows), problems


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print("=" * 78)
    print("BF16-GPTQ 델타: 차원별/토큰별 에너지 집중도 (모든 조합)")
    print("=" * 78)

    df, problems = run_all_combinations()

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"총 조합 수: {len(df)}")
    print(f"문제(파일 없음/shape 불일치): {len(problems)}건")
    for p in problems:
        print("  -", p)
    print(f"전체 결과 CSV 저장 위치: {OUTPUT_CSV_PATH}")
    print()

    print(df.to_string(index=False))
