import os
# OpenMP 중복 로드 오류 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import gc
import time
import logging
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
# [변경사항 1] auto_gptq -> gptqmodel
from gptqmodel import GPTQModel, QuantizeConfig

# ==========================================
# [기초 설정 1] 실험 재현성을 위한 시드(Seed) 고정
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# [기초 설정 2] 경로 및 연구용 디렉토리 변수 할당
# ==========================================

# 실행할 모델의 주석만 해제하여 사용하세요
# TARGET_MODEL = "TinyLlama"
# TARGET_MODEL = "Qwen2.5"
TARGET_MODEL = "Llama-3.2"

if TARGET_MODEL == "TinyLlama":
    model_path = "/mnt/c/Users/user/SLM/00_Base_Models/TinyLlama-1.1B-Chat-v1.0"
    MODEL_FAMILY = "TinyLlama_1.1B"
    TRAIN_METHOD = "Base_Scratch"

elif TARGET_MODEL == "Qwen2.5":
    model_path = "/mnt/c/Users/user/SLM/00_Base_Models/Qwen2.5-1.5B-Instruct"
    MODEL_FAMILY = "Qwen2.5_1.5B"
    TRAIN_METHOD = "Base_RLHF"

elif TARGET_MODEL == "Llama-3.2":
    model_path = "/mnt/c/Users/user/SLM/00_Base_Models/Llama-3.2-1B-Instruct"
    MODEL_FAMILY = "Llama_3.2_1B"
    TRAIN_METHOD = "Distillation"

# 공통 설정 경로(사용될 데이터셋 및 로그 저장 위치)
calib_dataset_path = "/mnt/c/Users/user/SLM/03_Evaluation_Datasets/wikitext-2-test.txt"
RESEARCH_ROOT = "/mnt/c/Users/user/SLM/02_cuda_aligned"
target_bits = [8, 4, 3, 2]

# ==========================================
# [기초 설정 3] 로깅(Logging) 시스템 구축 (모델 단위 실험 관리)
# ==========================================

# 1. 모델별 로그 디렉토리 생성
# - 실험 단위(모델 계열)별 로그를 구조적으로 분리하여 관리
# - 동일 모델에 대한 반복 실험 시 로그 추적 및 비교 용이
log_dir = os.path.join(RESEARCH_ROOT, "logs", MODEL_FAMILY)
os.makedirs(log_dir, exist_ok=True)

# 2. 마스터 로그 파일 정의
# - 전체 파이프라인 실행 과정을 단일 파일에 기록
# - 타임스탬프 기반으로 실행 단위(run-level) 로그를 구분
master_log_filename = os.path.join(
    log_dir, 
    f"master_log_{MODEL_FAMILY}_{TRAIN_METHOD}_{time.strftime('%Y%m%d_%H%M%S')}.log"
)

# 3. 로거(Logger) 설정
# - 파일 + 콘솔 출력 동시 기록
# - INFO 레벨 기준으로 실험 상태 추적
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# 파일 핸들러: 디스크에 영구 저장
master_file_handler = logging.FileHandler(master_log_filename, encoding='utf-8')
master_file_handler.setFormatter(formatter)
logger.addHandler(master_file_handler)

# 콘솔 핸들러: 실시간 출력 (디버깅 및 진행 확인 목적)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 4. 초기 실행 환경 로깅
# - 실험 재현성 확보를 위한 핵심 메타데이터 기록
logger.info(f"선택된 실험 모델: {MODEL_FAMILY} ({TRAIN_METHOD})")

# GPU 정보 기록 (CUDA 환경 검증 목적)
logger.info(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")

# 현재 시점의 GPU 메모리 할당량 - 로깅 시점 기준 할당된 메모리 상태
logger.info(f"초기 할당된 VRAM: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

# 5. 토크나이저 로드
# - 모델과 동일한 토크나이저를 사용하여 입력 일관성 유지
logger.info("-> 공통 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# # ==========================================
# [보조 함수] 디스크 용량 계산 헬퍼 함수
# ==========================================

# 디렉토리 전체 용량(MB) 계산 함수
# - 모델 크기 비교, 양자화 전후 용량 분석 등에 활용
def get_dir_size_mb(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

# ==========================================
# [기준값 초기화] (Baseline Metrics)
# ==========================================

# BF16 원본 모델 디스크 용량
# - 양자화 모델과의 저장 공간 비교 기준
baseline_disk_mb = get_dir_size_mb(model_path)

# BF16 기준 정적 VRAM 사용량
# - 이후 측정 시 비교 기준으로 활용 (초기값은 0으로 설정)
static_vram = 0.0

# ==========================================
# [Phase 1] BF16 대조군(Baseline) 측정
# ==========================================
logger.info("=" * 60)
logger.info(f"[Phase 1] 원본 모델 대조군 지표 측정 시작 ({MODEL_FAMILY} / {TRAIN_METHOD})")

try:
    logger.info("-> 원본 모델 VRAM 적재 중 (transformers.AutoModelForCausalLM)...")

    # 1) GPU 메모리 측정 초기화
    torch.cuda.reset_peak_memory_stats()
    
    # 2) BF16 모델 로드 및 GPU 적재
    precision_modes = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16, # torch_dtype을 dtype으로 변경
        low_cpu_mem_usage=True
    ).cuda()
    precision_modes.eval()

    # 3) 정적 VRAM 사용량 (모델 로드 직후)
    static_vram = torch.cuda.memory_allocated() / (1024**3)
    logger.info(f"-> [대조군 측정] 원본 모델 디스크 용량: {baseline_disk_mb:.2f} MB")
    logger.info(f"-> [대조군 측정] 순수 모델 적재 VRAM: {static_vram:.2f} GB ({static_vram * 1024:.2f} MB)")

    dummy_prompt = "The concept of quantization in neural networks is"

    # 4) 입력 데이터 준비 (attention_mask 포함)
    inputs = tokenizer(dummy_prompt, return_tensors="pt").to("cuda:0")
    
    logger.info("-> GPU 추론 웜업(Warm-up) 실행 중...")

    # 5) 워밍업 (커널 초기화 및 캐시 안정화)
    with torch.no_grad():
        precision_modes.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, # 명시적 전달 (경고 소거)
            max_new_tokens=10, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    logger.info("-> 추론 속도 및 피크 VRAM 측정 중...")
    max_new_tokens_to_generate = 128

    # 6) 추론 성능 측정 시작
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    # 7) 실제 추론 실행 (성능 측정 구간)
    with torch.no_grad():
        outputs = precision_modes.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, # 명시적 전달
            max_new_tokens=max_new_tokens_to_generate, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()

    # 8) Throughput 계산 (tokens/sec)
    generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    inference_time = end_time - start_time

    throughput = generated_tokens / inference_time

    # 9) 추론 중 최대 VRAM 사용량 (Peak)
    inference_peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    logger.info(f"-> [대조군 측정] 추론 속도 (Throughput): {throughput:.2f} tokens/sec")
    logger.info(f"-> [대조군 측정] 추론 시 최대 VRAM (Peak): {inference_peak_vram:.2f} GB")

except Exception as e:
    logger.error(f"-> [오류 발생] Phase 1 대조군 측정 실패: {e}")

finally:
    # 10) GPU 메모리 정리 (실험 간 간섭 방지)
    logger.info("-> Phase 1 종료. VRAM 할당 해제 및 캐시 초기화 진행...")
    if 'precision_modes' in locals():
        del precision_modes
    if 'inputs' in locals():
        del inputs
    if 'outputs' in locals():
        del outputs
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"-> 초기화 완료. 현재 잔여 VRAM: {torch.cuda.memory_allocated() / (1024**2):.2f} MB\n")

# ==========================================
# [Phase 2 준비] 캘리브레이션 데이터 로드
# ==========================================

def load_custom_calib_data(data_path, tokenizer_obj, n_samples=256, block_size=512):
    calib_data = []
    current_text = ""

    # 1) 텍스트 파일을 순차적으로 읽어 누적
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            current_text += line

            # 2) 충분한 길이가 되면 토크나이징 수행
            if len(current_text) > block_size * 4:
                encodings = tokenizer_obj(
                    current_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=block_size
                )

                # 3) 정확히 block_size 길이인 샘플만 사용 (입력 길이 통일)
                if encodings["input_ids"].shape[1] == block_size:
                    calib_data.append({
                        "input_ids": encodings["input_ids"],
                        "attention_mask": encodings["attention_mask"]
                    })
                    current_text = "" 

                # 4) 지정된 샘플 수 도달 시 종료
                if len(calib_data) >= n_samples:
                    break
    return calib_data

logger.info("=" * 60)

logger.info("[Phase 2 준비] 통제된 캘리브레이션 데이터 구축 중...")

# 5) 캘리브레이션 데이터 생성 실행
custom_calib_data = load_custom_calib_data(
    calib_dataset_path, 
    tokenizer, 
    n_samples=256, 
    block_size=512
)

# 6) 생성된 샘플 수 확인
logger.info(f"-> 캘리브레이션 샘플 {len(custom_calib_data)}개 로드 완료. (max_length=512)\n")

# ==========================================
# [Phase 2] GPTQ 양자화 파이프라인 (bit별 반복 실험)
# ==========================================
for w_bit in target_bits:

    # 1) 실험 결과 저장 경로 및 bit별 로그 설정
    quant_path = os.path.join(RESEARCH_ROOT, MODEL_FAMILY, TRAIN_METHOD, f"GPTQ_{w_bit}bit")
    os.makedirs(quant_path, exist_ok=True)
    
    # [추가됨] 해당 비트 폴더 내부에 분산 저장될 개별 로그 핸들러 추가
    bit_log_filename = os.path.join(quant_path, f"quant_log_{w_bit}bit.log")
    bit_file_handler = logging.FileHandler(bit_log_filename, encoding='utf-8')
    bit_file_handler.setFormatter(formatter)
    logger.addHandler(bit_file_handler) # 현재 비트 로그 핸들러 부착

    logger.info("=" * 60)
    logger.info(f"[{w_bit}-bit GPTQ 양자화 프로세스 시작]")
        
    # 2) 양자화 설정 정의 (bit, group_size 등 핵심 하이퍼파라미터)
    quantize_config = QuantizeConfig(
        bits=w_bit,
        group_size=128,
        desc_act=False,
        sym=True 
    )
    logger.info(f"-> 설정 적용: bits={w_bit}, group_size=128, desc_act=False, sym=True")
    logger.info("-> 원본 모델 VRAM 적재 중 (gptqmodel.GPTQModel)...")

    # 3) 모델 로드 (양자화 대상)
    model = GPTQModel.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        # low_cpu_mem_usage=True 제거 (gptqmodel이 자체적으로 메모리 관리 수행)
        dtype=torch.bfloat16 # torch_dtype을 dtype으로 변경
    )
    
    logger.info("-> 헤시안 행렬 계산 및 가중치 압축 진행 중...")

    # 4) 양자화 수행 (calibration 데이터 기반)
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    try:
        model.quantize(custom_calib_data)
        
        end_time = time.time()

        # 5) 양자화 시간 및 Peak VRAM 측정
        elapsed_time = end_time - start_time
        quant_peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        logger.info(f"-> [측정 결과] {w_bit}-bit 양자화 소요 시간: {elapsed_time:.2f}초")
        logger.info(f"-> [측정 결과] {w_bit}-bit 양자화 진행 중 최대 VRAM (Peak): {quant_peak_vram:.2f} GB")
        
        logger.info(f"-> {w_bit}-bit 모델 로컬 디스크 저장 중...")

        # 6) 양자화 모델 저장
        # 파라미터 제거 (gptqmodel은 기본적으로 safetensors로 저장함)
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)
        logger.info(f"-> 저장 완료: {quant_path}")
        
        # 7) 디스크 용량 감소율 계산 (Baseline 대비)
        quant_disk_mb = get_dir_size_mb(quant_path)
        disk_reduction_rate = (1 - (quant_disk_mb / baseline_disk_mb)) * 100
        logger.info(f"-> [효율 분석] {w_bit}-bit 디스크 용량: {quant_disk_mb:.2f} MB (대조군 대비 감소율: {disk_reduction_rate:.2f}%)")
        
        # 8) 메모리 정리 후 양자화 모델 재로딩 (VRAM 측정)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"-> VRAM 감소율 측정을 위해 {w_bit}-bit 양자화 모델 재적재 중...")
        q_model = GPTQModel.from_quantized(quant_path, device="cuda:0")
        
        # 9) 정적 VRAM 감소율 계산
        quant_vram_gb = torch.cuda.memory_allocated() / (1024**3)
        quant_vram_mb = quant_vram_gb * 1024
        
        if static_vram > 0:
            vram_reduction_rate = (1 - (quant_vram_gb / static_vram)) * 100
        else:
            vram_reduction_rate = 0.0
            
        logger.info(f"-> [효율 분석] {w_bit}-bit 적재 VRAM: {quant_vram_gb:.2f} GB ({quant_vram_mb:.2f} MB) (대조군 대비 감소율: {vram_reduction_rate:.2f}%)")
        logger.info(f"-> [효율 분석] {w_bit}-bit 모델 추론 속도 및 피크 VRAM 측정 중...")

        # 10) 추론 성능 측정 (Throughput + Peak VRAM)
        inputs = tokenizer(dummy_prompt, return_tensors="pt").to("cuda:0")
        
        # 워밍업
        with torch.no_grad():
            q_model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            
        torch.cuda.reset_peak_memory_stats()
        start_time_q = time.time()

        # 실제 추론
        with torch.no_grad():
            outputs_q = q_model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens_to_generate, pad_token_id=tokenizer.eos_token_id)
            
        end_time_q = time.time()
        
        # Throughput 계산
        generated_tokens_q = outputs_q.shape[1] - inputs.input_ids.shape[1]
        inference_time_q = end_time_q - start_time_q
        throughput_q = generated_tokens_q / inference_time_q

        # 추론 시 Peak VRAM
        inference_peak_vram_q = torch.cuda.max_memory_allocated() / (1024**3)

        logger.info(f"-> [효율 분석] {w_bit}-bit 추론 속도 (Throughput): {throughput_q:.2f} tokens/sec")
        logger.info(f"-> [효율 분석] {w_bit}-bit 추론 시 최대 VRAM (Peak): {inference_peak_vram_q:.2f} GB")
        
        del q_model
        
    except Exception as e:
        logger.error(f"-> [오류 발생] {w_bit}-bit 양자화 실패: {e}")
    
    finally:
        logger.info("-> VRAM 할당 해제 및 캐시 초기화 진행...")

        # 11) GPU 메모리 정리 (다음 실험 간 간섭 방지)
        if 'model' in locals():
            del model
        if 'q_model' in locals():
            del q_model
        if 'inputs' in locals():
            del inputs
        if 'outputs_q' in locals():
            del outputs_q

        torch.cuda.empty_cache()

        gc.collect()
        residual_vram = torch.cuda.memory_allocated() / (1024**2)
        logger.info(f"-> 초기화 완료. 현재 잔여 VRAM: {residual_vram:.2f} MB. 다음 프로세스로 넘어갑니다.\n")
        
        # 12) bit별 로그 핸들러 제거 (로그 분리 유지)
        logger.removeHandler(bit_file_handler)
        bit_file_handler.close()

# 전체 파이프라인 종료
logger.info("=" * 60)
logger.info("Phase 1(대조군 측정) 및 Phase 2(GPTQ 양자화 파이프라인)의 모든 실행이 정상 종료되었습니다.")