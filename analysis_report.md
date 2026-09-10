# SLM GPTQ 인지 붕괴 분석 레포트 (SLM GPTQ Collapse Analysis)

**작성 일시:** 2026년 9월 7일 13:58:45

---

## 1. 디렉토리 및 파일 구조

```text
D:\slm-gptq-collapse-analysis
├── .env
├── .git/
├── .gitignore
├── 00_Base_Models/                                 # 원본 언어 모델 저장소
│   ├── Llama-3.2-1B-Instruct/
│   ├── Qwen2.5-1.5B-Instruct/
│   ├── TinyLlama-1.1B-Chat-v1.0/
│   └── download_base_models.py                     # 모델 다운로드 스크립트
├── 01_cuda_misaligned/                             # CUDA 정렬 문제가 있었거나 구버전 환경의 실험 코드
│   ├── 01_Quantized_Models/
│   ├── 02_Experiment_Logs/
│   ├── 03_Extended Output/
│   ├── evaluation_results.csv
│   ├── evaluation_results_full_1.csv
│   ├── evaluation_results_full_2.csv
│   ├── llama_models.py                             # Llama 모델 양자화 스크립트 (gptqmodel 라이브러리 사용)
│   ├── qwen_models.py
│   └── tinyllama_models.py
├── 02_cuda_aligned/                                # CUDA 환경이 정렬된 메인 실험 및 평가 코드
│   ├── Experiment_Data_v1/
│   ├── Experiment_Data_v2/                         # V2: attention/MLP 등 모듈별 텐서 출력 저장 폴더
│   ├── Llama_3.2_1B/
│   ├── Qwen2.5_1.5B/
│   ├── TinyLlama_1.1B/
│   ├── __pycache__/
│   ├── all_logs/
│   ├── 01_quantize_pipeline_gptqmodel.py           # 통합 GPTQ 양자화 파이프라인
│   ├── 02_evaluate_perplexity_gptq.py              # 단일 모델 Perplexity(PPL) 평가 스크립트
│   ├── 03_auto_evaluate_ppl.py                     # 다중 모델 PPL 자동 평가 스크립트
│   ├── 04_baseline_ppl_evaluation.py               # 원본 모델(BF16) PPL 베이스라인 평가
│   ├── 04_runtime_log_bf16.txt                     # BF16 모델 실행 로그
│   ├── 04_runtime_log_quant.txt                    # 양자화 모델 실행 로그
│   ├── 05_controlled_inference_eval.py             # 다중 프롬프트를 통한 통제된 추론 평가(인지 붕괴 분석) 스크립트
│   └── runtime_log_full.txt
├── 03_Evaluation_Datasets/                         # 모델 평가용 데이터셋
│   ├── cache/
│   ├── results_cognitive/
│   ├── eval_datasets.py                            # 데이터셋 다운로드 및 전처리 (WikiText-2, HellaSwag)
│   ├── hellaswag_eval.py                           # HellaSwag 상식 추론 평가 스크립트
│   ├── runtime_log_hellaswag.txt
│   ├── wikitext-2-test.txt                         # PPL 평가 및 Calibration 데이터셋 (Raw Text)
│   └── wikitext-2-valid.txt
└── 04_llm_quant_analysis/                          # 실험 결과 데이터 분석 및 시각화를 위한 Jupyter Notebook 및 이미지
    ├── 01_static_analysis.ipynb                    # 정적 분석 (VRAM 메모리 점유율 등)
    ├── 02_dynamic_analysis.ipynb                   # 동적 분석
    ├── 03_performance_analysis.ipynb               # 성능 분석
    ├── 04_ppl_quantization_visualization.ipynb     # 비트별 양자화 PPL 변화 시각화
    ├── 05_plot_cognitive_collapse.ipynb            # 인지 붕괴(Cognitive Collapse) 현상 시각화
    ├── 06_analyze_p1_framework.ipynb               # 프롬프트 1 (다단계 절차 실행) 결과 분석
    ├── 07_analyze_p2_framework.ipynb               # 프롬프트 2 (사실적 지식 인출) 결과 분석
    ├── 08_analyze_p3_framework.ipynb               # 프롬프트 3 (포맷 구조 유지) 결과 분석
    ├── 09_analyze_p4_framework.ipynb               # 프롬프트 4 (논리적 모순 처리) 결과 분석
    ├── Fig2_Internal_Error_Pivot_Academic.png      # 논문 또는 학술 보고용 시각화 차트 이미지들
    ├── Fig2_P1_PPL_Correlation_Clean.png
    ├── Fig2_P1_PPL_Correlation_Clean_Final.png
    ├── Fig2_P1_PPL_Correlation_Paradox.png
    ├── Fig2_P1_PPL_Correlation_Single_MarkerAlert.png
    ├── Fig2_P1_PPL_Correlation_Single_Segmented.png
    ├── Fig2_P1_Qwen_PPL_Correlation_Paradox.png
    └── defense_ppl_graph_matplotlib.png
```

---

## 2. 연구 주제 및 핵심 목적
이 프로젝트는 **10억(1B) 개 내외의 파라미터를 가진 소형 언어 모델(SLM, Small Language Models)에 GPTQ 양자화를 적용했을 때 발생하는 '인지 붕괴(Cognitive Collapse)' 현상을 분석하는 연구**입니다.
* 언어 모델을 16-bit(BF16)에서 8-bit, 4-bit, 3-bit, 2-bit로 압축(양자화)해 나갈 때 언어적 능력(Perplexity), 상식(HellaSwag), 그리고 세부적인 인지 능력들이 어떤 비트 수준에서, 어떤 형태로 붕괴되는지 계량화하고 그 원인을 모듈 단위에서 추적하는 것을 목적으로 합니다.
* 기존 `auto_gptq` 라이브러리 대신 `gptqmodel` 라이브러리를 채택하여 파이프라인이 구축되어 있습니다.

---

## 3. 연구 대상 모델 (Target Models)
연구에 사용된 SLM(Small Language Model)은 학습 방식과 아키텍처가 다른 3가지 모델입니다.
1. **Llama-3.2-1B-Instruct**: Distillation(증류) 방식으로 학습된 모델
2. **Qwen2.5-1.5B-Instruct**: Base 모델에 RLHF(인간 피드백 기반 강화학습)를 거친 모델
3. **TinyLlama-1.1B-Chat-v1.0**: 밑바닥부터(Base_Scratch) 학습된 초소형 모델

---

## 4. 사용된 데이터셋 및 측정 기준
* **WikiText-2 (`wikitext-2-test.txt`)**: GPTQ 양자화 진행 시 캘리브레이션(Calibration) 용도 및 일반적인 언어 모델링 성능 지표인 Perplexity(PPL)를 측정하기 위해 사용.
* **HellaSwag**: 모델의 기본적 상식 추론(Common Sense Reasoning) 능력을 측정하기 위한 데이터셋.

---

## 5. 인지 붕괴 측정 프롬프트 분석 (Controlled Inference Evaluation)
`05_controlled_inference_eval.py`와 `04_llm_quant_analysis`의 분석 노트북을 통해 4가지 축(P1 ~ P4)으로 통제된 프롬프트를 주입하여 인지 능력을 평가합니다. 
(출력 데이터는 원본 BF16 계산의 충실성을 확인하기 위해 모델의 `residual stream`, `attn_output`, `mlp_output` 등을 float32 `.pt` 파일 형태로 저장합니다.)

* **Prompt 1 (P1): 다단계 절차를 순서대로 실행하는 능력 (Multi-step Procedural Execution)**
  * 산술 절차(예: 3 * 4 * 5 를 계산한 뒤 그 결과에 (2 * X) + 20 수식 적용)를 통한 절차적 기억과 실행 능력을 측정.
  * **[예외 사항]**: TinyLlama-1.1B는 원본 BF16 상태에서도 이 다단계 산술을 수행하지 못해 베이스라인 자체가 붕괴되어 있으므로 P1 분석 대상에서 자동 제외하도록 설정됨.
* **Prompt 2 (P2): 사실적 지식 인출 능력 (Factual Recall & Knowledge Probing)**
  * 1B 체급 모델에서 복잡한 제약조건을 주면 Attention 희석(Attention Dilution)이 발생해 점수가 하락하므로 프롬프트 조건을 단순화하여 정보 인출 능력 자체만 평가.
* **Prompt 3 (P3): 포맷 구조 및 구문 제약 유지 (Format Structure / Syntax constraints)**
  * 출력 형태를 지정했을 때 이를 잊지 않고 얼마나 정확하게 준수하는지 평가.
* **Prompt 4 (P4): 논리적 모순 처리 및 패러독스 핸들링 (Logical Contradiction / Paradox handling)**

---

## 6. 진행된 분석 시각화 내용 (`04_llm_quant_analysis` 기반)
* **정적 분석 (Static Analysis)**: 베이스 모델(BF16)과 각 양자화 모델(8, 4, 3, 2-bit)이 차지하는 VRAM 메모리 점유율(GiB) 비교 및 Embedding, LM Head 등 양자화 예외 레이어가 차지하는 비중 분석.
* **동적/성능 분석 (Dynamic & Performance)**: 추론 과정에서의 리소스 사용량 및 속도 변화.
* **PPL vs 양자화 비트 시각화**: 16bit -> 8bit -> 4bit -> 3bit -> 2bit 로 낮아짐에 따라 3개 모델의 Perplexity 점수가 어떻게 훼손(상승)되는지 Plotly 산점도와 꺾은선 그래프로 분석.
* **프레임워크 개별 분석**: P1, P2, P3, P4 각 영역에서 어떤 형태의 오류(에러)가 발생하는지에 대해 집중 분석한 결과물을 다양한 형태의 논문(학술)용 `.png` 차트(예: 에러 피벗, P1과 PPL간의 상관관계 패러독스 등)로 도출.
