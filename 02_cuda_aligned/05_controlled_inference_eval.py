import os
import json
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ==============================================================================
# 1. 고정 변수 및 환경 설정
# ==============================================================================
BASE_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # 02_cuda_aligned의 상위 폴더
BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, "00_Base_Models")
QUANT_MODELS_DIR = SCRIPT_DIR                            # 양자화 모델은 스크립트와 동일 폴더
SAVE_BASE_DIR = os.path.join(SCRIPT_DIR, "Experiment_Data_v2")

# ==============================================================================
# [v2 추가] 모듈별 출력(attention / MLP) 저장 토글
#   - True  : residual stream(기존) + attn_output + mlp_output 모두 저장 (분석 풀셋)
#   - 저장 dtype은 BF16 원본 충실성 유지를 위해 기존과 동일하게 float32로 .pt 저장.
#     (요청: BF16 계산 충실성 + 가능한 모든 텐서 저장)
# ==============================================================================
SAVE_MODULE_OUTPUTS = True

EXPERIMENT_CONFIGS = [
    {
        "base_model_dir": "Llama-3.2-1B-Instruct",
        "quant_base_dir": "Llama_3.2_1B",
        "quant_sub_dir": "Distillation",
    },
    {
        "base_model_dir": "Qwen2.5-1.5B-Instruct",
        "quant_base_dir": "Qwen2.5_1.5B",
        "quant_sub_dir": "Base_RLHF",
    },
    {
        "base_model_dir": "TinyLlama-1.1B-Chat-v1.0",
        "quant_base_dir": "TinyLlama_1.1B",
        "quant_sub_dir": "Base_Scratch",
    },
]

QUANT_METHODS = ["GPTQ"]
BIT_LEVELS = ["2bit", "3bit", "4bit", "8bit"]

PROMPTS_DATASET = [
    # ----------------------------------------------------------------------
    # Prompt 1: Multi-step Procedural Execution
    #
    # 측정 축: "다단계 절차를 순서대로 실행하는 능력" (절차적 기억 + 단계 실행).
    #          P2(지식 인출)·P3(포맷 구조)·P4(모순 처리)와 직교하도록 설계.
    #
    # 산술 절차 — 3*4*5, (2*60)+20 = 140
    #   - 곱셈/대입이라는 인지 부하가 있는 절차.
    #   - BF16 검증 완료(2026-05): Llama-3.2-1B, Qwen2.5-1.5B 모두 수준 4.
    #
    # [TinyLlama-1.1B-Chat-v1.0 제외]
    #   TinyLlama는 어떤 형태의 다단계 절차 실행도 BF16에서 불가(3*4*5를 12로
    #   계산하거나 지시문을 그대로 복창). 베이스라인이 무너지면 양자화 손상을
    #   측정할 수 없으므로 P1에서 TinyLlama를 제외.
    #   → PROMPT_EXCLUDE_MODELS[0] 에 정의됨 (실행 루프가 자동 스킵).
    #   → 모델별 차등 프롬프트(분기 dict)는 불필요해져 단순 list로 정리됨.
    # ----------------------------------------------------------------------
    [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence that solves mathematical problems "
                "through precise step-by-step reasoning. You must follow this EXACT format without exception:\n\n"
                "Step 1: Multiply the three given numbers to find the total number of existing books. "
                "Show the multiplication explicitly. Example format: 'A * B * C = result'.\n"
                "Step 2: Apply the formula (2 * result_from_step_1) + 20 to find the number of new books. "
                "Show the calculation explicitly.\n"
                "Final Answer: <OUTPUT ONLY THE FINAL NUMBER. NO WORDS, NO UNITS, NO PUNCTUATION.>"
            ),
        },
        {
            "role": "user",
            "content": (
                "Calculate how many NEW books the librarian adds based on these exact conditions:\n"
                "- Condition 1: The library has 3 floors.\n"
                "- Condition 2: Each floor contains exactly 4 bookshelves.\n"
                "- Condition 3: Each bookshelf holds exactly 5 books.\n"
                "- Condition 4: The number of new books added equals (2 * total number of existing books) + 20."
            ),
        },
    ],

    # ----------------------------------------------------------------------
    # Prompt 2: Factual Recall & Knowledge Probing  (롤백 + 단순화)
    #
    # 이전 강화 조항("MUST explicitly include BOTH dates AND ALL three names")
    # 이 1B 체급에서 어텐션 희석 유발(Qwen 점수 하락) → 단순한 원본으로 롤백.
    # Conclusion 통합 여부는 LLM 채점기가 후처리 항목으로 측정.
    # 2026-05 미세조정: Step 1 날짜 분리를 "two separate bullet points"로 명시
    #   (TinyLlama가 두 날짜를 한 문장에 합치는 포맷 위반 측정을 명확화).
    #   강한 대문자 강제는 피하고, 포맷 힌트 수준으로만 적용해 어텐션 희석 방지.
    # ----------------------------------------------------------------------
    [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence that provides verified historical facts. "
                "Use the following format exactly:\n\n"
                "Step 1 (Dates): State the launch date and the lunar landing date "
                "as two separate bullet points.\n"
                "Step 2 (Crew): State the full name (first name and last name) of each of the three astronauts.\n"
                "Conclusion: One concise summary paragraph integrating the above facts."
            ),
        },
        {
            "role": "user",
            "content": "Describe the verified historical facts of the Apollo 11 mission.",
        },
    ],

    # ----------------------------------------------------------------------
    # Prompt 3: Format Adherence (formerly IFEval-style)  (전면 재설계)
    #
    # 이전 미시 통사 제약(시작어 통일, 쉼표 금지, 정확 N문장)은 자연어 확률
    # 분포와 정면 충돌하여 1B 체급 모두 베이스라인 미달.
    # → 거시 포맷 제약(Markdown 헤더 + 번호 리스트)으로 전환.
    # 1B 모델도 학습 시 마크다운을 풍부하게 본 데이터이므로 안정적 통과.
    # 양자화 시 마크다운 토큰(##, 1., 2., ...)이 깨지는 양상 관찰 가능.
    # ----------------------------------------------------------------------
    [
        {
            "role": "system",
            "content": (
                "You write structured technical answers. Your response must follow this exact "
                "Markdown format:\n\n"
                "## Summary\n"
                "<one sentence>\n\n"
                "## Key Points\n"
                "1. <first point>\n"
                "2. <second point>\n"
                "3. <third point>\n\n"
                "Output exactly two headers (Summary, Key Points) and exactly three numbered items. "
                "Do not include any other text outside this structure."
            ),
        },
        {
            "role": "user",
            "content": "Explain how ocean tides are influenced by the moon.",
        },
    ],

    # ----------------------------------------------------------------------
    # Prompt 4: Adversarial Stress Test  (변경 없음)
    #
    # BF16 graceful degradation 정상. 양자화 시 토큰 누수/무한 반복으로
    # 무너지는 임계점 측정용. P4 전용 루브릭 적용.
    # ----------------------------------------------------------------------
    [
        {
            "role": "system",
            "content": (
                "You must obey ALL of the following rules simultaneously and without "
                "any exception:\n"
                "Rule 1: Your entire response must consist of exactly one sentence "
                "containing no more than 10 words in total.\n"
                "Rule 2: Your response must also contain a numbered list with at least "
                "5 separate items.\n"
                "Rule 3: Each item in the numbered list must itself be a complete "
                "paragraph of at least 3 sentences.\n"
                "Rule 4: You must satisfy Rules 1, 2, and 3 above completely and at "
                "the same time."
            ),
        },
        {
            "role": "user",
            "content": "Explain the process of photosynthesis in plants.",
        },
    ],
]


def resolve_prompt_for_model(prompt_entry, base_model_dir: str) -> list:
    """
    PROMPTS_DATASET[i]가 모델별 분기 dict인 경우 해당 모델의 메시지를 반환,
    단순 list인 경우 그대로 반환.

    [2026-05 현황] 현재 4개 프롬프트 모두 단순 list (모델 차등 불필요).
    TinyLlama의 P1·P3 제외는 PROMPT_EXCLUDE_MODELS로 처리되므로, 이전의
    P1 모델별 분기 dict는 제거되었음. 이 함수는 향후 모델별 차등이 다시
    필요해질 경우를 대비해 유지하며, list는 부작용 없이 그대로 통과시킴.

    분기 우선순위 (dict인 경우):
      1. base_model_dir와 정확 일치하는 키
      2. "default" 키
      3. 둘 다 없으면 ValueError
    """
    if isinstance(prompt_entry, list):
        return prompt_entry  # 모든 모델 공통 프롬프트
    if isinstance(prompt_entry, dict):
        if base_model_dir in prompt_entry:
            return prompt_entry[base_model_dir]
        if "default" in prompt_entry:
            return prompt_entry["default"]
        raise ValueError(
            f"프롬프트 분기 dict에서 '{base_model_dir}' 모델용 키도 'default'도 없습니다."
        )
    raise TypeError(f"PROMPTS_DATASET 항목 타입이 잘못됨: {type(prompt_entry)}")


# ==============================================================================
# 프롬프트별 모델 제외 매핑
# ==============================================================================
# BF16 베이스라인 검증(2026-05) 결과:
#   - P1(절차 실행), P3(포맷 준수): TinyLlama-1.1B는 BF16에서 수준 1 (지시 수행
#     능력 자체 결여 — 지시문 복창 / 마크다운 완전 무시). 베이스라인이 무너지면
#     양자화 손상을 측정할 수 없으므로 해당 조합 제외.
#   - P2(지식 인출): TinyLlama 수준 3 (사실 정확) → 포함.
#   - P4(적대적): TinyLlama 수준 1이지만 P4는 BF16 붕괴 자체가 측정 대상 → 포함.
#
# 인덱스는 PROMPTS_DATASET의 0-based 위치. (0=P1, 1=P2, 2=P3, 3=P4)
PROMPT_EXCLUDE_MODELS = {
    0: ["TinyLlama-1.1B-Chat-v1.0"],   # P1: 절차 실행 — TinyLlama 제외
    2: ["TinyLlama-1.1B-Chat-v1.0"],   # P3: 포맷 준수 — TinyLlama 제외
}


def is_prompt_excluded_for_model(prompt_index: int, base_model_dir: str) -> bool:
    """해당 (프롬프트 인덱스, 모델) 조합이 제외 대상인지 판정."""
    return base_model_dir in PROMPT_EXCLUDE_MODELS.get(prompt_index, [])



def reset_environment(seed: int = BASE_SEED) -> None:
    """가비지 컬렉션, CUDA 캐시 정리, 시드 재고정."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# [v2 추가] 모듈별(attention / MLP) 출력 캡처용 forward hook
# ==============================================================================
def locate_decoder_layers(model):
    """
    Llama-3.2 / Qwen2.5 / TinyLlama 는 모두 표준 구조를 따른다:
        model.model.layers[i].self_attn   (어텐션 블록, o_proj 직후 출력)
        model.model.layers[i].mlp         (MLP 블록, down_proj 직후 출력)
    구조가 다르면 즉시 멈춰서 잘못된 위치에 hook 거는 것을 방지한다.
    반환: layers (nn.ModuleList)
    """
    decoder = getattr(model, "model", model)
    if not hasattr(decoder, "layers"):
        raise RuntimeError(
            "디코더 레이어 경로(model.model.layers)를 찾을 수 없습니다. "
            "print(model) 로 구조를 확인하고 locate_decoder_layers를 수정하세요."
        )
    layers = decoder.layers
    sample = layers[0]
    if not (hasattr(sample, "self_attn") and hasattr(sample, "mlp")):
        raise RuntimeError(
            "레이어에 self_attn / mlp 속성이 없습니다. 실제 모듈명을 확인하세요: "
            f"{[n for n, _ in sample.named_children()]}"
        )
    return layers


def register_module_capture_hooks(model):
    """
    각 디코더 레이어의 self_attn 출력과 mlp 출력을 forward hook으로 가로채 저장.
    hook은 '읽기 전용'으로 텐서를 detach→cpu→float32 복사만 하므로
    모델 연산·시드·생성 결과에 일절 영향을 주지 않는다(재현성 보존).
    반환: (captured, handles)
       captured["attn"][i] : i번째 디코더 레이어 어텐션 출력 [1, seq, hidden]
       captured["mlp"][i]  : i번째 디코더 레이어 MLP 출력      [1, seq, hidden]
    """
    captured = {"attn": {}, "mlp": {}}
    handles = []
    layers = locate_decoder_layers(model)

    def make_hook(store, idx):
        def hook(module, inputs, output):
            # self_attn은 (hidden, attn_weights, past_kv) 튜플을 반환할 수 있음.
            # mlp는 보통 텐서. 양쪽 모두 첫 요소가 hidden state.
            t = output[0] if isinstance(output, tuple) else output
            store[idx] = t.detach().to("cpu", dtype=torch.float32)
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(captured["attn"], i)))
        handles.append(layer.mlp.register_forward_hook(make_hook(captured["mlp"], i)))
    return captured, handles


def build_teacher_forcing_mask(generated_ids: torch.Tensor,
                               prompt_len: int,
                               eos_token_id: int,
                               pad_token_id: int) -> torch.Tensor:
    """
    Teacher Forcing용 attention mask 생성.

    [핵심 원칙]
    - prompt 영역(0 ~ prompt_len-1): 무조건 1
    - 생성 영역에서 첫 번째 EOS 위치까지(포함): 1
    - 첫 EOS 이후의 진짜 padding 영역만: 0

    [왜 이렇게 하나]
    Llama 계열은 pad_token이 없어서 pad_token_id = eos_token_id로 설정됨.
    이 상태에서 단순히 (ids != pad_token_id) 마스크를 만들면, 모델이
    정상 종료를 알리며 출력한 마지막 EOS 토큰까지 마스킹되어 hidden state
    분석에서 가장 중요한 "생성 종료 시점"이 사라지는 치명적 결함 발생.

    [동작 보장]
    - batch_size=1 가정 (현 스크립트의 실험 설정)
    - generate()가 EOS로 정상 종료한 경우: 마지막 토큰만 EOS, 그 외엔 1
    - generate()가 max_new_tokens 도달로 잘린 경우: 모든 토큰 1
    - reference 시퀀스를 양자화 모델 평가에 재사용하는 경우에도 동일하게 동작
    """
    assert generated_ids.dim() == 2 and generated_ids.size(0) == 1, \
        f"이 함수는 batch_size=1 텐서를 가정합니다. shape={tuple(generated_ids.shape)}"

    seq_len = generated_ids.size(1)
    mask = torch.ones_like(generated_ids, dtype=torch.long)

    if eos_token_id is None:
        return mask  # EOS 없으면 보호할 것도 없음

    # 생성 영역(prompt_len 이후)에서 첫 EOS 위치 탐색
    gen_part = generated_ids[0, prompt_len:]
    eos_positions = (gen_part == eos_token_id).nonzero(as_tuple=False)

    if eos_positions.numel() == 0:
        # 생성 영역에 EOS 없음 → max_new_tokens로 잘렸거나 silent collapse
        # 모든 토큰을 attention 대상으로 둠
        return mask

    first_eos_in_gen = eos_positions[0, 0].item()
    first_eos_abs = prompt_len + first_eos_in_gen  # 시퀀스 전체 기준 절대 위치

    # 첫 EOS 다음 토큰부터 끝까지를 0으로 (진짜 padding 영역)
    if first_eos_abs + 1 < seq_len:
        mask[:, first_eos_abs + 1:] = 0

    return mask


def detect_termination_reason(generated_ids: torch.Tensor,
                              prompt_len: int,
                              max_new_tokens: int,
                              eos_token_id: int) -> str:
    """
    생성 종료 사유를 분류하여 메타데이터로 기록.
    Silent Collapse(루브릭에 없는 회색 지대) 추적용.

    Returns:
        "natural_eos"     : 정상 EOS로 종료
        "max_tokens"      : max_new_tokens 한계 도달
        "silent_collapse" : EOS 없이 한계 전에 멈춤(이론상 발생 불가지만 안전망)
    """
    if eos_token_id is None:
        return "no_eos_token_defined"

    gen_len = generated_ids.size(1) - prompt_len
    last_token = generated_ids[0, -1].item()

    if last_token == eos_token_id:
        return "natural_eos"
    if gen_len >= max_new_tokens:
        return "max_tokens"
    return "silent_collapse"


# ==============================================================================
# 2. 단일 모델 평가 함수 (메모리 로드/해제 캡슐화)
# ==============================================================================
def evaluate_model(evaluate_original_model: bool, config: dict,
                   quant_method: str = None, bit_level: str = None) -> None:
    base_model_dir = config["base_model_dir"]
    quant_base_dir = config["quant_base_dir"]
    quant_sub_dir = config["quant_sub_dir"]

    # 1) 경로 및 실험 ID 설정
    if evaluate_original_model:
        raw_model_path = os.path.join(BASE_MODELS_DIR, base_model_dir)
        experiment_id = f"{base_model_dir}_Original_BF16"
        print(f"\n{'='*60}\n[*] 모드: 원본 모델(Base) 평가 시작 - {base_model_dir}\n{'='*60}")
    else:
        raw_model_path = os.path.join(
            QUANT_MODELS_DIR, quant_base_dir, quant_sub_dir,
            f"{quant_method}_{bit_level}",
        )
        experiment_id = f"{base_model_dir}_{quant_method}_{bit_level}"
        print(f"\n{'='*60}\n[*] 모드: 양자화 모델 평가 시작 ({quant_method} {bit_level}) - {base_model_dir}\n{'='*60}")

    model_path = os.path.abspath(raw_model_path)
    if not os.path.exists(model_path):
        print(f"[!] 경고: 지정된 경로를 찾을 수 없어 건너뜁니다: {model_path}")
        return

    # 2) 모델 및 토크나이저 로드 (try 블록 바깥에서 None으로 선언 → finally에서 안전한 해제)
    model = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError(f"{base_model_dir}의 토크나이저에 EOS 토큰이 없습니다.")

        # 원본은 BF16 명시, 양자화 모델은 FP16(GPTQ dequant 표준) 사용
        target_dtype = torch.bfloat16 if evaluate_original_model else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=target_dtype,
            trust_remote_code=True,
        )
        model.eval()
        actual_dtype = next(model.parameters()).dtype

        # 3) 프롬프트 순회
        for idx, prompt_entry in enumerate(PROMPTS_DATASET):
            prompt_id = f"Prompt_{idx+1:02d}"

            # 베이스라인 부적합 조합 스킵 (예: TinyLlama의 P1·P3)
            if is_prompt_excluded_for_model(idx, base_model_dir):
                print(f"  -> [{idx+1}/{len(PROMPTS_DATASET)}] {prompt_id} "
                      f"SKIP (베이스라인 부적합: {base_model_dir} × {prompt_id})")
                continue

            # 프롬프트 메시지 결정 (현재 4개 모두 단순 list; 헬퍼는 그대로 통과)
            messages = resolve_prompt_for_model(prompt_entry, base_model_dir)
            print(f"  -> [{idx+1}/{len(PROMPTS_DATASET)}] 프롬프트 처리 중...")
            reset_environment(BASE_SEED)

            current_save_dir = os.path.join(SAVE_BASE_DIR, experiment_id, prompt_id)
            tensors_dir = os.path.join(current_save_dir, "tensors")
            os.makedirs(tensors_dir, exist_ok=True)

            # 채팅 템플릿 적용 (System/User 역할 분리 그대로 투입)
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            reference_dir = os.path.join(SAVE_BASE_DIR, f"{base_model_dir}_Original_BF16", prompt_id)
            reference_ids_path = os.path.join(reference_dir, "reference_input_ids.pt")

            generation_kwargs = {
                "do_sample": False,
                "repetition_penalty": 1.0,
                # 350 → 512 (2026-05): P4(적대적+광합성) TinyLlama가 한도에서 잘려
                # EOS 못 보고 종료 → silent_collapse 위양성 발생. 정상 종료 기회 확보.
                "max_new_tokens": 512,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

            # ----------------------------------------------------------------
            # 분기별 처리: full_generated_ids 를 일관되게 구성
            # ----------------------------------------------------------------
            termination_reason = None  # 메타데이터에 기록할 변수

            if evaluate_original_model:
                # 원본 모델: 직접 생성 → reference 저장
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs,
                    )
                full_generated_ids = outputs

                os.makedirs(reference_dir, exist_ok=True)
                torch.save(full_generated_ids.detach().cpu(), reference_ids_path)

                generated_ids = full_generated_ids[0][input_ids.shape[-1]:]

                # ── 텍스트 저장: 정성 평가용(clean)과 토큰 누수 검증용(raw) 두 버전 ──
                # clean: skip_special_tokens=True → 사람이 읽고 루브릭 채점할 텍스트
                generated_text_clean = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                # raw: skip_special_tokens=False → 진짜 토큰 누수(수준 0) 판별 시
                #     사용. 정상 종료 EOS는 시퀀스 끝에 1개만 있어야 정상.
                generated_text_raw = tokenizer.decode(
                    generated_ids, skip_special_tokens=False
                )
                with open(os.path.join(current_save_dir, "generated_output.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(generated_text_clean)
                with open(os.path.join(current_save_dir, "generated_output_raw.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(generated_text_raw)

                # 종료 사유 분류 (silent collapse 추적)
                termination_reason = detect_termination_reason(
                    full_generated_ids,
                    prompt_len=input_ids.shape[-1],
                    max_new_tokens=generation_kwargs["max_new_tokens"],
                    eos_token_id=tokenizer.eos_token_id,
                )

                # generated_text 변수 통일(루프 끝 메모리 정리 호환)
                generated_text = generated_text_clean
                del outputs, generated_text_clean, generated_text_raw

            else:
                # 양자화 모델
                if not os.path.exists(reference_ids_path):
                    raise FileNotFoundError(
                        f"Reference 시퀀스가 없습니다. Base 모델을 먼저 실행해야 합니다: {reference_ids_path}"
                    )

                # (1) 정성 평가용: 양자화 모델의 자체 텍스트 생성
                with torch.no_grad():
                    quant_outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs,
                    )
                quant_generated_ids = quant_outputs[0][input_ids.shape[-1]:]

                # ── 양자화 모델의 자체 출력도 clean / raw 두 버전으로 저장 ──
                # 수준 0(토큰 누수) 판별의 결정적 증거는 raw 파일에서 확인.
                # clean 파일은 수준 1~4 채점에 사용.
                quant_text_clean = tokenizer.decode(
                    quant_generated_ids, skip_special_tokens=True
                )
                quant_text_raw = tokenizer.decode(
                    quant_generated_ids, skip_special_tokens=False
                )
                with open(os.path.join(current_save_dir, "quant_generated_output.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(quant_text_clean)
                with open(os.path.join(current_save_dir, "quant_generated_output_raw.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(quant_text_raw)

                # 양자화 모델의 종료 사유 (정성 평가의 핵심 신호)
                termination_reason = detect_termination_reason(
                    quant_outputs,
                    prompt_len=input_ids.shape[-1],
                    max_new_tokens=generation_kwargs["max_new_tokens"],
                    eos_token_id=tokenizer.eos_token_id,
                )
                del quant_outputs, quant_generated_ids
                del quant_text_clean, quant_text_raw

                # (2) 정량 평가용: Reference 시퀀스 로드 → Teacher Forcing
                full_generated_ids = torch.load(
                    reference_ids_path, map_location=model.device
                )
                # 명시적 long 캐스팅(혹시 저장 dtype 다를 경우 대비)
                full_generated_ids = full_generated_ids.long()

                # ─── 안전장치: 양자화 모델 토크나이저가 만든 prompt 부분이
                #     원본 모델 reference의 prompt 부분과 비트 단위로 일치하는지 검증.
                #     불일치 시 슬라이싱 인덱스가 어긋나 hidden state 분석이 무의미해짐.
                prompt_len = input_ids.shape[-1]
                assert full_generated_ids.shape[-1] > prompt_len, (
                    f"Reference 시퀀스 길이({full_generated_ids.shape[-1]})가 "
                    f"prompt 길이({prompt_len})보다 짧거나 같습니다. "
                    f"Reference 파일이 손상되었을 수 있습니다."
                )
                ref_prompt_part = full_generated_ids[0, :prompt_len]
                cur_prompt_part = input_ids[0].to(ref_prompt_part.device)
                if not torch.equal(ref_prompt_part, cur_prompt_part):
                    raise RuntimeError(
                        f"[{experiment_id} / {prompt_id}] 토크나이저 불일치 감지: "
                        f"양자화 모델의 prompt 토큰이 원본 모델 reference의 prompt 토큰과 "
                        f"일치하지 않습니다. 양자화 폴더의 tokenizer/chat_template을 "
                        f"확인해주세요."
                    )

                generated_ids = full_generated_ids[0][prompt_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                with open(os.path.join(current_save_dir, "reference_text_used.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(generated_text)

            # ----------------------------------------------------------------
            # 공통: Hidden States 추출 (Teacher Forcing 1-pass)
            # ----------------------------------------------------------------
            # [핵심 수정] 기존의 (ids != pad_token_id) 마스크는 pad_token == eos_token
            # 인 환경(Llama 등)에서 정상 종료의 마지막 EOS 토큰까지 어텐션에서
            # 배제시켜 hidden state 분석을 변질시켰음. 첫 EOS 위치까지(포함)는
            # 1로 보호하는 헬퍼 함수로 대체.
            tf_attention_mask = build_teacher_forcing_mask(
                generated_ids=full_generated_ids,
                prompt_len=input_ids.shape[-1],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ).to(model.device)

            with torch.no_grad():
                # [v2] forward 직전에 모듈 hook 부착 (읽기 전용, 재현성 무영향)
                if SAVE_MODULE_OUTPUTS:
                    captured, hook_handles = register_module_capture_hooks(model)
                else:
                    captured, hook_handles = None, []

                forward_outputs = model(
                    full_generated_ids,
                    attention_mask=tf_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # [v2] forward 직후 즉시 hook 해제 (다음 프롬프트로 누수 방지)
                for _h in hook_handles:
                    _h.remove()

            hidden_states = forward_outputs.hidden_states
            tensor_metadata = []
            # ── Layer-wise 분석용 통계 (사진의 06 분석을 효율화) ──
            # 원본 텐서는 그대로 저장하므로 사후 L2/Cosine 계산은 가능하지만,
            # Rogue Dimension(소수 차원이 norm을 지배) 식별과 L2-Norm 교차검증을
            # 분석 단계에서 거대 텐서 재로드 없이 바로 할 수 있도록
            # 차원별/레이어별 요약 통계를 추론 시점에 함께 저장.
            layer_statistics = []

            for layer_idx, layer_tensor in enumerate(hidden_states):
                tensor_cpu = layer_tensor.detach().cpu().to(torch.float32)
                file_name = f"layer_{layer_idx}_hidden_states.pt"
                torch.save(tensor_cpu, os.path.join(tensors_dir, file_name))
                tensor_metadata.append({
                    "layer": layer_idx,
                    "shape": list(tensor_cpu.shape),
                    "file_name": file_name,
                })

                # tensor_cpu shape: [batch=1, seq_len, hidden_dim]
                t = tensor_cpu[0]  # [seq_len, hidden_dim]
                # 토큰별 L2 norm (시퀀스 위치별 표현 크기) — silent collapse/outlier 추적
                token_l2 = t.norm(dim=-1)  # [seq_len]
                # 차원별 통계 (Rogue Dimension 식별의 핵심)
                #   - dim_mean_abs: 각 차원의 평균 절대값
                #   - dim_l2: 각 차원의 L2 norm (시퀀스 축 기준)
                dim_mean_abs = t.abs().mean(dim=0)   # [hidden_dim]
                dim_l2 = t.norm(dim=0)               # [hidden_dim]
                # Rogue Dimension 후보: norm 기여도 상위 10개 차원
                topk = min(10, dim_l2.shape[0])
                top_vals, top_idx = torch.topk(dim_l2, topk)
                # 전체 norm 대비 상위 10개 차원이 차지하는 비율
                # (이 값이 크면 소수 차원이 표현을 지배 = Rogue Dimension 강함)
                total_sq = (dim_l2 ** 2).sum()
                top_sq = (top_vals ** 2).sum()
                rogue_ratio = (top_sq / total_sq).item() if total_sq > 0 else 0.0

                layer_statistics.append({
                    "layer": layer_idx,
                    # 레이어 전체 표현 크기
                    "global_l2_norm": float(t.norm().item()),
                    "global_mean_abs": float(t.abs().mean().item()),
                    "global_max_abs": float(t.abs().max().item()),
                    # 토큰 위치별 L2 (마지막 토큰 = 생성 결정 지점)
                    "token_l2_mean": float(token_l2.mean().item()),
                    "token_l2_max": float(token_l2.max().item()),
                    "token_l2_last": float(token_l2[-1].item()),
                    # Rogue Dimension 진단
                    "rogue_top10_energy_ratio": float(rogue_ratio),
                    "rogue_top10_dim_indices": [int(i) for i in top_idx.tolist()],
                    "dim_max_abs_value": float(dim_mean_abs.max().item()),
                    "dim_max_abs_index": int(dim_mean_abs.argmax().item()),
                })

                # ── [v2] 모듈별(attention/MLP) 출력 저장 ──
                # hidden_states는 [embedding, dec0_out, dec1_out, ...] 순서이므로
                # hidden_states[layer_idx]의 디코더 레이어 인덱스는 (layer_idx - 1).
                # layer_idx == 0 은 임베딩 출력이라 대응 모듈이 없음 → 건너뜀.
                if SAVE_MODULE_OUTPUTS and captured is not None:
                    dec_idx = layer_idx - 1
                    if dec_idx >= 0 and dec_idx in captured["attn"]:
                        attn_t = captured["attn"][dec_idx]   # [1, seq, hidden]
                        mlp_t = captured["mlp"][dec_idx]
                        # 인덱스 정합 검증: 모듈 출력 shape == residual stream shape
                        # (한 칸이라도 어긋나면 분석 전체가 무의미해지므로 강하게 검증)
                        assert attn_t.shape == tensor_cpu.shape, (
                            f"[{experiment_id}/{prompt_id}] attn shape {tuple(attn_t.shape)} "
                            f"!= hidden {tuple(tensor_cpu.shape)} (dec_idx={dec_idx})"
                        )
                        assert mlp_t.shape == tensor_cpu.shape, (
                            f"[{experiment_id}/{prompt_id}] mlp shape {tuple(mlp_t.shape)} "
                            f"!= hidden {tuple(tensor_cpu.shape)} (dec_idx={dec_idx})"
                        )
                        torch.save(attn_t, os.path.join(
                            tensors_dir, f"layer_{dec_idx}_attn_output.pt"))
                        torch.save(mlp_t, os.path.join(
                            tensors_dir, f"layer_{dec_idx}_mlp_output.pt"))
                        tensor_metadata.append({
                            "decoder_layer": dec_idx,
                            "attn_file": f"layer_{dec_idx}_attn_output.pt",
                            "mlp_file": f"layer_{dec_idx}_mlp_output.pt",
                            "maps_to_hidden_index": layer_idx,
                        })
                        # 모듈별 요약 통계도 함께 (분석 시 텐서 재로드 없이 활용)
                        ta = attn_t[0]
                        tm = mlp_t[0]
                        layer_statistics.append({
                            "module_for_decoder_layer": dec_idx,
                            "attn_global_l2_norm": float(ta.norm().item()),
                            "attn_token_l2_last": float(ta.norm(dim=-1)[-1].item()),
                            "mlp_global_l2_norm": float(tm.norm().item()),
                            "mlp_token_l2_last": float(tm.norm(dim=-1)[-1].item()),
                        })
                        del attn_t, mlp_t, ta, tm

                del tensor_cpu, t, token_l2, dim_mean_abs, dim_l2

            # 레이어 통계는 별도 JSON으로 저장 (분석 스크립트가 텐서 없이 바로 사용)
            with open(os.path.join(current_save_dir, "layer_statistics.json"),
                      "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_id": experiment_id,
                    "model_name": base_model_dir,
                    "bit_level": "16bit" if evaluate_original_model else bit_level,
                    "prompt_id": prompt_id,
                    "note": (
                        "Per-layer activation statistics for layer-wise L2 / "
                        "Rogue-Dimension cross-validation (Timkey 2021, Ethayarajh 2019). "
                        "rogue_top10_energy_ratio close to 1.0 means a few dimensions "
                        "dominate the representation norm — cosine similarity on such "
                        "layers is contaminated and must be cross-checked with L2."
                    ),
                    "layers": layer_statistics,
                }, f, indent=4, ensure_ascii=False)

            with open(os.path.join(current_save_dir, "tensor_metadata.json"),
                      "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_id": experiment_id,
                    "is_original_model": evaluate_original_model,
                    "model_name": base_model_dir,
                    "quantization": "None" if evaluate_original_model else quant_method,
                    "bit_level": "16bit" if evaluate_original_model else bit_level,
                    "prompt_id": prompt_id,
                    "seed_used": BASE_SEED,
                    "dtype_used": str(actual_dtype),
                    "phase": "Full Sequence (Teacher Forcing)",
                    "input_sequence_length": int(input_ids.shape[-1]),
                    "full_sequence_length": int(full_generated_ids.shape[-1]),
                    "total_layers_extracted": len(hidden_states),
                    # ── 분석 단계 보조 정보 ──
                    # termination_reason: "natural_eos" | "max_tokens" | "silent_collapse"
                    #   → silent_collapse가 곧 자료2-③에서 지적된 회색 지대.
                    #     루브릭상 수준 1과 수준 3 사이에서 분석가가 판단할 신호.
                    "termination_reason": termination_reason,
                    "tf_mask_active_tokens": int(tf_attention_mask.sum().item()),
                    "tf_mask_total_tokens": int(tf_attention_mask.numel()),
                    "tokenizer_pad_eq_eos": (
                        tokenizer.pad_token_id == tokenizer.eos_token_id
                    ),
                    "tensors": tensor_metadata,
                    "prompt_content": messages,
                }, f, indent=4, ensure_ascii=False)

            # 프롬프트 루프 메모리 정리: 분기별로 정의된 변수만 안전하게 해제
            # (다음 iteration 시작 시 reset_environment가 다시 호출되므로 여기선 메모리만 해제)
            del inputs, input_ids, attention_mask
            del full_generated_ids, generated_ids, generated_text
            del forward_outputs, hidden_states, tf_attention_mask
            if captured is not None:
                captured["attn"].clear()
                captured["mlp"].clear()
                del captured
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[*] {experiment_id} 데이터 수집 완료")

    finally:
        # ======================================================================
        # 단일 모델 평가 종료 시 VRAM 완전 해제
        # ======================================================================
        print(f"[*] {experiment_id} 모델 메모리 해제 진행")
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# 3. 자동화 실행 루프
# ==============================================================================
if __name__ == "__main__":
    # ---------------- 경로 사전 검증 (잘못된 경로면 즉시 종료) ----------------
    print("[설정] 경로 정보:")
    print(f"  SCRIPT_DIR       = {SCRIPT_DIR}")
    print(f"  PROJECT_ROOT     = {PROJECT_ROOT}")
    print(f"  BASE_MODELS_DIR  = {BASE_MODELS_DIR}")
    print(f"  QUANT_MODELS_DIR = {QUANT_MODELS_DIR}")
    print(f"  SAVE_BASE_DIR    = {SAVE_BASE_DIR}")

    if not os.path.isdir(BASE_MODELS_DIR):
        raise FileNotFoundError(
            f"BASE_MODELS_DIR가 존재하지 않습니다: {BASE_MODELS_DIR}\n"
            f"본 스크립트는 PROJECT_ROOT/02_cuda_aligned/ 내부에 위치해야 하며, "
            f"PROJECT_ROOT/00_Base_Models/ 가 존재해야 합니다."
        )
    if not os.path.isdir(QUANT_MODELS_DIR):
        raise FileNotFoundError(f"QUANT_MODELS_DIR가 존재하지 않습니다: {QUANT_MODELS_DIR}")

    # 각 모델별 base / quantized 폴더가 실제 존재하는지 미리 점검
    print("\n[설정] 폴더 존재 여부 사전 확인:")
    for cfg in EXPERIMENT_CONFIGS:
        base_p = os.path.join(BASE_MODELS_DIR, cfg["base_model_dir"])
        print(f"  [Base ] {'OK ' if os.path.isdir(base_p) else 'MISS'} -> {base_p}")
        for q_method in QUANT_METHODS:
            for b_level in BIT_LEVELS:
                q_p = os.path.join(
                    QUANT_MODELS_DIR, cfg["quant_base_dir"], cfg["quant_sub_dir"],
                    f"{q_method}_{b_level}",
                )
                print(f"  [Quant] {'OK ' if os.path.isdir(q_p) else 'MISS'} -> {q_p}")
    print()

    # 프롬프트×모델 제외 조합 안내 (베이스라인 부적합으로 스킵되는 항목)
    if PROMPT_EXCLUDE_MODELS:
        print("[설정] 베이스라인 부적합으로 제외되는 (프롬프트 × 모델) 조합:")
        for p_idx, excluded in sorted(PROMPT_EXCLUDE_MODELS.items()):
            for m in excluded:
                print(f"  - Prompt_{p_idx+1:02d} × {m}  (BF16 수준 1, 측정 불가)")
        print()

    # ---------------- 실제 평가 루프 ----------------
    for cfg in EXPERIMENT_CONFIGS:
        # 단계 1: Base 원본 모델 평가 (Reference 시퀀스 생성용)
        evaluate_model(evaluate_original_model=True, config=cfg)

        # 단계 2: 해당 모델의 양자화 버전 순차 평가
        for q_method in QUANT_METHODS:
            for b_level in BIT_LEVELS:
                evaluate_model(
                    evaluate_original_model=False,
                    config=cfg,
                    quant_method=q_method,
                    bit_level=b_level,
                )

    print("\n[✔] 모든 모델 및 프롬프트에 대한 자동화 평가가 완료되었습니다.")