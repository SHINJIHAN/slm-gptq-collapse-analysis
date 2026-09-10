"""
12_massive_activation_verification.py

목적: layer 1에서 발견된 "고정 차원"(양자화 오차 에너지가 프롬프트가 바뀌어도 같은
hidden dimension에 몰리는 현상)이 진짜 massive activation(Sun et al. 2024)인지
확정하기 위한 마지막 검증 두 가지.

  [검증 1] 그 차원들이 "원본 BF16 텐서 자체"에서 다른 차원보다 압도적으로 큰가?
           (지금까지는 "양자화 오차가 그 차원에 몰린다"만 봤음. 원본 자체가 이미
           거대해야 "양자화가 massive activation을 건드린다"는 인과가 완성됨.)

  [핵심 그래프] layer 1의 차원별 오차 에너지(delta**2를 토큰 축으로 합산)를
           x=차원 인덱스, y=에너지인 막대로 그리고, 4개 프롬프트를 겹쳐 그린다.
           히스토그램(값의 분포)이 아니라 "차원 인덱스별" 막대라는 점이 핵심 —
           프롬프트가 달라져도 같은 x 위치에서 막대가 서면 그게 massive activation의
           input-invariance다. bit level별로 보고 싶어서 4개 bit level(8/4/3/2bit)을
           서브플롯으로 나누고, 각 서브플롯 안에 4개 프롬프트를 겹쳐 그린다(어느 하나만
           골라서 보여주지 않기 위함).

"고정 차원" 후보는 임의로 고르지 않고, layer 1에서 (4개 bit level x 4개 프롬프트 =
16개) top-5 dim 집합 전체에 걸쳐 나온 빈도를 세어, 16번 모두(=입력도, 양자화 강도도
안 가리고 항상) 등장한 차원만 "고정 차원"으로 확정한다. 그 기준을 만족하는 차원이
없으면 없다고 그대로 보고한다(억지로 후보를 만들지 않음).

기존 파일(05_controlled_inference_eval.py, layer_statistics.json,
10_compute_true_quantization_error.py, layer_delta_statistics.json,
11_dimension_token_concentration_analysis.py, dimension_token_concentration_full.csv,
모든 분석 노트북)은 전혀 건드리지 않는다. 이 스크립트는 새로 읽기만 하고, 새 출력
파일(PNG 4장 + 콘솔 표)만 만든다. 표는 요약하지 않고 발견된 고정 차원 전부를 그대로
보여준다.
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 그래프에 쓰는 한글(제목/축/범례)이 matplotlib 기본 폰트(DejaVu Sans)에는 없어서
# 그대로 두면 글자가 네모(missing glyph)로 깨진다. 기존 분석 노트북들과 동일하게
# Windows 한글 폰트로 강제 지정(마이너스 기호 깨짐 방지 옵션도 함께).
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

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
COMPARISON_BITS = ["GPTQ_8bit", "GPTQ_4bit", "GPTQ_3bit", "GPTQ_2bit"]  # 정밀도 감소 순
PROMPTS = ["Prompt_01", "Prompt_02", "Prompt_03", "Prompt_04"]
BLOCK_TYPES = ["attn", "mlp"]

TARGET_LAYER = 1
TOPK = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "massive_activation_figs")
os.makedirs(FIG_DIR, exist_ok=True)

# dataviz 스킬 검증된 팔레트 slot 1/2/3/7 (light mode, all-pairs CVD/normal-vision 통과.
# slot4 yellow는 slot2 orange와 all-pairs 기준 충돌하는 것으로 문서화되어 있어 제외하고
# slot7 violet으로 대체 — node scripts/validate_palette.js로 재검증 완료).
PROMPT_COLORS = {
    "Prompt_01": "#2a78d6",  # blue
    "Prompt_02": "#eb6834",  # orange
    "Prompt_03": "#1baf7a",  # aqua
    "Prompt_04": "#4a3aa7",  # violet
}
PROMPT_MARKERS = {
    "Prompt_01": "o",
    "Prompt_02": "s",
    "Prompt_03": "^",
    "Prompt_04": "D",
}

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_TEXT = "#898781"
PRIMARY_TEXT = "#0b0b0b"


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
# [검증 1] 원본 BF16 자체에서 차원별 활성화 크기
# ==============================================================================
def compute_bf16_dim_stats(model_name, prompt_id, layer_idx, block_type):
    """
    원본 BF16 텐서(양자화와 무관, delta 아님) 하나에서 차원별 통계를 계산.
    반환: DataFrame, index=dim, columns=[mean_abs, max_abs, l2, rank_by_mean_abs,
    percentile_by_mean_abs]  (hidden_dim 전체에 대해 계산)
    """
    bf16_t, path = load_module_tensor(model_name, REFERENCE_BIT, prompt_id, layer_idx, block_type)
    if bf16_t is None:
        return None

    t = bf16_t.float().squeeze(0)  # [seq, dim]
    dim_mean_abs = t.abs().mean(0).numpy()
    dim_max_abs = t.abs().max(0).values.numpy()
    dim_l2 = t.norm(dim=0).numpy()

    hidden_dim = t.shape[1]
    order = np.argsort(-dim_mean_abs)  # 내림차순: rank 1 = 가장 큰 mean_abs
    rank = np.empty(hidden_dim, dtype=int)
    rank[order] = np.arange(1, hidden_dim + 1)
    percentile = 100.0 * (1.0 - (rank - 1) / hidden_dim)  # rank=1 -> 100, rank=hidden_dim -> ~0

    return pd.DataFrame({
        "dim": np.arange(hidden_dim),
        "bf16_mean_abs": dim_mean_abs,
        "bf16_max_abs": dim_max_abs,
        "bf16_l2": dim_l2,
        "bf16_rank_by_mean_abs": rank,
        "bf16_percentile_by_mean_abs": percentile,
    }).set_index("dim")


# ==============================================================================
# 델타 기반 차원별 에너지 (핵심 그래프 + 후보 차원 탐색용)
# ==============================================================================
def compute_dim_energy(model_name, bit_level, prompt_id, layer_idx, block_type):
    """
    dim_energy[i] = sum over tokens of delta[:, i]**2   (delta = bf16 - gptq)
    반환: numpy array [hidden_dim], 또는 실패 시 None
    """
    bf16_t, bf16_path = load_module_tensor(model_name, REFERENCE_BIT, prompt_id, layer_idx, block_type)
    gptq_t, gptq_path = load_module_tensor(model_name, bit_level, prompt_id, layer_idx, block_type)

    if bf16_t is None or gptq_t is None:
        return None
    if tuple(bf16_t.shape) != tuple(gptq_t.shape):
        return None

    delta = (bf16_t.float() - gptq_t.float()).squeeze(0)  # [seq, dim]
    sq = delta ** 2
    dim_energy = sq.sum(0).numpy()  # [dim]
    return dim_energy


def find_fixed_dims(model_name, block_type, layer_idx, topk=TOPK):
    """
    (bit_level x prompt) 16개 조합 각각에서 layer_idx의 dim_energy top-k를 구하고,
    그 dim이 16개 조합 중 몇 번 top-k에 들었는지 빈도를 센다.
    반환: (freq_df, dim_energy_by_combo)
      freq_df: 최소 1번이라도 top-k에 든 모든 dim에 대한 빈도표 (전부 표시, 잘라내지 않음)
      dim_energy_by_combo: {(bit_level, prompt_id): dim_energy array} — 그래프용으로 재사용
    """
    counter = Counter()
    membership = {}  # dim -> list of (bit_level, prompt_id) it appeared in
    dim_energy_by_combo = {}

    for bit_level in COMPARISON_BITS:
        for prompt_id in PROMPTS:
            dim_energy = compute_dim_energy(model_name, bit_level, prompt_id, layer_idx, block_type)
            dim_energy_by_combo[(bit_level, prompt_id)] = dim_energy
            if dim_energy is None:
                continue
            top_idx = np.argsort(-dim_energy)[:topk]
            for d in top_idx:
                d = int(d)
                counter[d] += 1
                membership.setdefault(d, []).append(f"{bit_level}/{prompt_id}")

    n_combos = len(COMPARISON_BITS) * len(PROMPTS)
    rows = []
    for dim, freq in counter.items():
        rows.append({
            "model": model_name,
            "block_type": block_type,
            "layer": layer_idx,
            "dim": dim,
            "topk_frequency": freq,
            "topk_frequency_out_of": n_combos,
            "appeared_in": ",".join(membership[dim]),
        })
    freq_df = pd.DataFrame(rows).sort_values(
        ["topk_frequency", "dim"], ascending=[False, True]
    ).reset_index(drop=True)

    return freq_df, dim_energy_by_combo


# ==============================================================================
# 핵심 그래프: layer 1 차원별 오차 에너지, bit level x 프롬프트
# ==============================================================================
def plot_dimension_energy(model_name, block_type, layer_idx, dim_energy_by_combo, fixed_dims, hidden_dim):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"{model_name} — layer {layer_idx} {block_type} 출력, BF16-GPTQ 델타 에너지 "
        f"(차원 인덱스별, 프롬프트 4개 겹쳐 표시)",
        fontsize=13, fontweight="bold", color=PRIMARY_TEXT,
    )

    for ax, bit_level in zip(axes.flat, COMPARISON_BITS):
        global_max_per_dim = np.zeros(hidden_dim)

        for prompt_id in PROMPTS:
            dim_energy = dim_energy_by_combo.get((bit_level, prompt_id))
            if dim_energy is None:
                continue
            global_max_per_dim = np.maximum(global_max_per_dim, dim_energy)
            ax.bar(
                np.arange(len(dim_energy)), dim_energy,
                width=1.0, color=PROMPT_COLORS[prompt_id], alpha=0.55,
                edgecolor="none", label=prompt_id,
            )

        ax.set_yscale("log")
        # 기본 log 포매터는 지수를 mathtext(예: 10^-2)로 그리는데, 유니코드 마이너스
        # 글리프가 Malgun Gothic에 없어 깨진다. 순수 ASCII 문자열 포매터로 대체해
        # 폰트와 무관하게 항상 정상 렌더링되도록 한다.
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0e}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(bit_level, fontsize=11, color=PRIMARY_TEXT, loc="left")
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.tick_params(colors=MUTED_TEXT)

        # 겹쳐 그린 막대는 같은 x에 완전히 포개져서, 4개 중 실제로 몇 개가 거기서
        # 같이 튀는 건지 눈으로 구분이 안 된다(맨 위에 그려진 계열 색으로 덮임).
        # "고정 차원"으로 확정된 곳만 x를 살짝 벌려 프롬프트별 마커를 따로 찍어서,
        # 하나가 아니라 4개 프롬프트 전부가 그 자리에서 같이 크다는 것을 직접 보여준다.
        jitter = np.array([-6, -2, 2, 6])
        for d in fixed_dims:
            if d >= hidden_dim:
                continue
            for offset, prompt_id in zip(jitter, PROMPTS):
                dim_energy = dim_energy_by_combo.get((bit_level, prompt_id))
                if dim_energy is None:
                    continue
                val = dim_energy[d]
                if val <= 0:
                    continue
                ax.scatter(
                    [d + offset], [val], marker=PROMPT_MARKERS[prompt_id],
                    s=32, color=PROMPT_COLORS[prompt_id], edgecolor="white",
                    linewidth=0.6, zorder=6,
                )

        # 선택적 direct label: "고정 차원"으로 확정된 것만, 그 서브플롯에서의
        # 최고 높이 위에 인덱스 표시 (모든 막대에 라벨을 달지 않음)
        for d in fixed_dims:
            if d < hidden_dim and global_max_per_dim[d] > 0:
                ax.annotate(
                    str(d),
                    xy=(d, global_max_per_dim[d]),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=PRIMARY_TEXT,
                )

    axes[0, 0].set_ylabel("차원별 오차 에너지 Σ(Δ²) (log scale)", color=PRIMARY_TEXT)
    axes[1, 0].set_ylabel("차원별 오차 에너지 Σ(Δ²) (log scale)", color=PRIMARY_TEXT)
    axes[1, 0].set_xlabel("hidden dimension index", color=PRIMARY_TEXT)
    axes[1, 1].set_xlabel("hidden dimension index", color=PRIMARY_TEXT)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PROMPT_COLORS[p], alpha=0.55) for p in PROMPTS
    ]
    fig.legend(handles, PROMPTS, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = os.path.join(FIG_DIR, f"{model_name}_{block_type}_layer{layer_idx}_dim_energy.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ==============================================================================
# 실행
# ==============================================================================
if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    all_freq_tables = []
    all_bf16_check_tables = []
    saved_figs = []

    for model_name, num_layers in MODELS.items():
        for block_type in BLOCK_TYPES:
            print("=" * 78)
            print(f"[모델={model_name} block={block_type} layer={TARGET_LAYER}]")
            print("=" * 78)

            # --- 후보(고정) 차원 탐색: 16개(bit x prompt) 조합의 top-5 빈도 ---
            freq_df, dim_energy_by_combo = find_fixed_dims(model_name, block_type, TARGET_LAYER, topk=TOPK)
            freq_df.insert(0, "target_layer", TARGET_LAYER)
            all_freq_tables.append(freq_df)

            n_combos = len(COMPARISON_BITS) * len(PROMPTS)
            print(f"\n[빈도표] top-{TOPK}에 한 번이라도 든 모든 차원 (총 {n_combos}개 조합 중 등장 횟수) — 잘라내지 않고 전부 표시:")
            print(freq_df.to_string(index=False))

            fixed_dims = freq_df.loc[freq_df["topk_frequency"] == n_combos, "dim"].tolist()
            if fixed_dims:
                print(f"\n[고정 차원 확정] {n_combos}개 조합 전부(모든 bit level x 모든 프롬프트)에서 "
                      f"top-{TOPK}에 든 차원: {fixed_dims}")
            else:
                max_freq = freq_df["topk_frequency"].max() if not freq_df.empty else 0
                fixed_dims = freq_df.loc[freq_df["topk_frequency"] == max_freq, "dim"].tolist()
                print(f"\n[고정 차원 확정 실패] {n_combos}개 조합 전부에서 공통으로 나온 차원 없음. "
                      f"대신 최고 빈도({max_freq}/{n_combos})를 보인 차원을 참고용으로 표시: {fixed_dims}")

            # --- 검증 1: 그 차원들이 원본 BF16 자체에서도 압도적으로 큰가 ---
            hidden_dim_for_model = None
            bf16_rows = []
            for prompt_id in PROMPTS:
                stats_df = compute_bf16_dim_stats(model_name, prompt_id, TARGET_LAYER, block_type)
                if stats_df is None:
                    continue
                hidden_dim_for_model = len(stats_df)
                for d in fixed_dims:
                    if d not in stats_df.index:
                        continue
                    row = stats_df.loc[d]
                    bf16_rows.append({
                        "model": model_name,
                        "block_type": block_type,
                        "layer": TARGET_LAYER,
                        "prompt": prompt_id,
                        "dim": d,
                        "bf16_mean_abs": row["bf16_mean_abs"],
                        "bf16_max_abs": row["bf16_max_abs"],
                        "bf16_l2": row["bf16_l2"],
                        "rank_by_mean_abs": int(row["bf16_rank_by_mean_abs"]),
                        "hidden_dim_total": hidden_dim_for_model,
                        "percentile_by_mean_abs": row["bf16_percentile_by_mean_abs"],
                    })
            bf16_check_df = pd.DataFrame(bf16_rows)
            all_bf16_check_tables.append(bf16_check_df)

            print(f"\n[검증 1] 고정 차원 {fixed_dims}이 원본 BF16 자체에서 차원별 크기 순위 상 어디에 있는가 "
                  f"(hidden_dim={hidden_dim_for_model}개 중 rank 1 = 가장 큰 차원, 프롬프트별로 각각 표시, 평균 내지 않음):")
            if not bf16_check_df.empty:
                print(bf16_check_df.to_string(index=False))
            else:
                print("  (고정 차원 없음 또는 BF16 데이터 없음 — 표 생략)")

            # --- 핵심 그래프 ---
            if hidden_dim_for_model is not None:
                out_path = plot_dimension_energy(
                    model_name, block_type, TARGET_LAYER, dim_energy_by_combo,
                    fixed_dims, hidden_dim_for_model,
                )
                saved_figs.append(out_path)
                print(f"\n[그래프 저장] {out_path}")
            print()

    print("=" * 78)
    print("전체 요약 (요약 통계 아님 — 저장된 산출물 목록만)")
    print("=" * 78)
    print(f"저장된 그래프: {len(saved_figs)}개")
    for p in saved_figs:
        print("  -", p)

    freq_all_df = pd.concat(all_freq_tables, ignore_index=True) if all_freq_tables else pd.DataFrame()
    bf16_all_df = pd.concat(all_bf16_check_tables, ignore_index=True) if all_bf16_check_tables else pd.DataFrame()

    freq_csv = os.path.join(FIG_DIR, "fixed_dim_frequency_table.csv")
    bf16_csv = os.path.join(FIG_DIR, "fixed_dim_bf16_magnitude_check.csv")
    freq_all_df.to_csv(freq_csv, index=False, encoding="utf-8-sig")
    bf16_all_df.to_csv(bf16_csv, index=False, encoding="utf-8-sig")
    print(f"빈도표 전체 CSV: {freq_csv}  ({len(freq_all_df)}행)")
    print(f"BF16 크기 검증 전체 CSV: {bf16_csv}  ({len(bf16_all_df)}행)")
