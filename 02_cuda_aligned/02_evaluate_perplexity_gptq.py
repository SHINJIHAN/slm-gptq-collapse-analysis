import os
import gc
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel

# ==========================================
# 1. 환경 및 경로 설정 (WSL 환경 기준)
# ==========================================
BASE_DIR = "/mnt/c/Users/user/SLM"
DATASET_PATH = os.path.join(BASE_DIR, r"03_Evaluation_Datasets/wikitext-2-valid.txt")
CSV_LOG_PATH = os.path.join(BASE_DIR, r"02_cuda_aligned/all_logs/evaluation_results.csv")

# ==========================================
# 2. PPL(Perplexity) 측정 핵심 함수
# ==========================================
def calculate_perplexity(model, tokenizer, text_path, stride=512, max_length=2048):
    """
    Sliding Window 방식을 사용한 정밀한 PPL 측정 함수
    """
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 전체 텍스트 토큰화
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    # max_length 제한 내에서 stride만큼 이동하며 손실(Loss) 계산
    for begin_loc in tqdm(range(0, seq_len, stride), desc="PPL Calculation"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        # 컨텍스트 부분의 target은 손실 계산에서 제외 (-100)
        target_ids[:, :-trg_len] = -100 

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break

    # 평균 NLL을 지수화하여 PPL 산출
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# ==========================================
# 3. 모델 평가 및 안전 관리 함수 (VRAM 누수 방지)
# ==========================================
def evaluate_model(model_path, model_type, dataset_path, log_csv_path):
    print(f"\n[INFO] 로딩 시작: {model_path} (Type: {model_type})")
    model = None
    tokenizer = None
    
    try:
        # 분기 처리: 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        # 분기 처리: 모델 로드 방식
        if model_type == "base":
            # 원본 모델은 FP16으로 로드하여 Baseline 측정
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
        elif model_type == "gptq":
            # 양자화 모델은 GPTQModel 사용
            model = GPTQModel.from_quantized(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            raise ValueError("model_type은 'base' 또는 'gptq'여야 합니다.")
            
        model.eval()
        
        # PPL 측정 수행
        print("[INFO] Perplexity 측정 중...")
        ppl_value = calculate_perplexity(model, tokenizer, dataset_path)
        print(f"[RESULT] 도출된 PPL: {ppl_value:.4f}")
        
        # CSV 실시간 기록 (Data Loss 방지)
        df = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
            "model_type": model_type,
            "perplexity": ppl_value
        }])
        
        # 파일이 없으면 헤더 포함 생성, 있으면 데이터만 추가
        file_exists = os.path.isfile(log_csv_path)
        df.to_csv(log_csv_path, mode='a', header=not file_exists, index=False)
        print(f"[INFO] 결과 저장 완료: {log_csv_path}")
        
    except Exception as e:
        print(f"[ERROR] 평가 중 오류 발생 ({model_path}): {e}")
        
    finally:
        # VRAM 강제 초기화 (메모리 파편화 및 OOM 방지)
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
            
        torch.cuda.empty_cache()
        gc.collect()
        print("[INFO] VRAM 캐시 클리어 완료.\n")

# ==========================================
# 4. 단일 테스트 실행 (Hardcoding)
# ==========================================
if __name__ == "__main__":
    # 향후 전체 파이프라인 가동 시, 아래 딕셔너리 리스트를 디렉토리 순회 로직으로 대체
    test_queue = [
        {
            "path": os.path.join(BASE_DIR, r"02_cuda_aligned/TinyLlama_1.1B/Base_Scratch/GPTQ_8bit"),
            "type": "gptq"
        }
        # 예시: Base 모델 평가 시 아래 주석 해제 및 경로 수정
        # ,{
        #     "path": os.path.join(BASE_DIR, r"00_Base_Models/TinyLlama-1.1B-Base"),
        #     "type": "base"
        # }
    ]
    
    # CSV 저장 폴더가 없다면 생성
    os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)
    
    for task in test_queue:
        evaluate_model(
            model_path=task["path"],
            model_type=task["type"],
            dataset_path=DATASET_PATH,
            log_csv_path=CSV_LOG_PATH
        )