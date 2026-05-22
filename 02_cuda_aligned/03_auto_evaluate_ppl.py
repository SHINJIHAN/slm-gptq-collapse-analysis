import os
import gc
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel

# ==========================================
# 1. 환경 및 경로 설정 (WSL 환경 기준)
# ==========================================
BASE_DIR = Path("/mnt/c/Users/user/SLM")
MODELS_DIR = BASE_DIR / "02_cuda_aligned"
DATASET_PATH = BASE_DIR / "03_Evaluation_Datasets" / "wikitext-2-valid.txt"
CSV_LOG_PATH = MODELS_DIR / "all_logs" / "evaluation_results_full.csv"

# ==========================================
# 2. PPL(Perplexity) 측정 핵심 함수
# ==========================================
def calculate_perplexity(model, tokenizer, text_path, stride=512, max_length=2048):
    """Sliding Window 방식을 사용한 정밀한 PPL 측정 함수"""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    target_lengths = [] # 각 스텝의 타겟 길이를 저장할 리스트 추가
    prev_end_loc = 0        
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="   [BASE] 측정 중", leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # outputs.loss는 현재 배치(trg_len)의 평균 NLL입니다.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood * trg_len) # 평균 Loss에 타겟 길이를 곱해 총 NLL을 구함
        target_lengths.append(trg_len)            # 연산에 사용된 실제 타겟 토큰 수 저장
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # (총 NLL의 합) / (총 타겟 토큰 수) -> 전체 텍스트에 대한 정확한 토큰당 평균 Loss
    avg_nll = torch.stack(nlls).sum() / sum(target_lengths)
    ppl = torch.exp(avg_nll)
    
    return ppl.item()

# ==========================================
# 3. 모델 평가 및 안전 관리 모듈
# ==========================================
def evaluate_model(model_path, model_type, dataset_path, log_csv_path):
    model = None
    tokenizer = None
    ppl_value = None
    
    try:
        # 분기 1: 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
        
        # 분기 2: 아키텍처에 따른 모델 로드
        if model_type == "base":
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), device_map="auto", torch_dtype=torch.bfloat16
            )
        elif model_type == "gptq":
            model = GPTQModel.from_quantized(
                str(model_path), device_map="auto", trust_remote_code=True
            )
        else:
            raise ValueError("Unknown model_type")
            
        model.eval()
        
        # 분기 3: PPL 연산
        ppl_value = calculate_perplexity(model, tokenizer, str(dataset_path))
        
        return ppl_value

    except Exception as e:
        print(f"\n   [ERROR] 평가 실패: {e}")
        return None
        
    finally:
        # ★ 가장 중요한 VRAM 강제 초기화 구역 ★
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        
        torch.cuda.empty_cache()
        gc.collect()

# ==========================================
# 4. 동적 경로 수집 및 자동화 파이프라인
# ==========================================
def build_evaluation_queue(models_dir):
    """디렉토리를 순회하여 평가할 모델 목록을 자동 생성"""
    queue = []
    print(f"[*] '{models_dir.name}' 폴더 내의 모델을 스캔합니다...")
    
    # model.safetensors 파일이 있는 모든 폴더를 타겟으로 지정
    safetensor_files = list(models_dir.rglob("model.safetensors"))
    
    for sf in safetensor_files:
        model_dir = sf.parent
        dir_name = model_dir.name
        
        # 경로 파싱 (예: Llama_3.2_1B/Distillation/GPTQ_8bit)
        try:
            model_family = model_dir.parent.parent.name # Llama_3.2_1B
            condition = model_dir.parent.name           # Distillation
            bit_level = dir_name                        # GPTQ_8bit
        except:
            # 폴더 깊이가 다를 경우 대비 (안전 장치)
            model_family = "Unknown"
            condition = "Unknown"
            bit_level = dir_name

        # GPTQ 모델인지 원본(Base) 모델인지 판별
        m_type = "gptq" if "GPTQ" in dir_name else "base"
        
        queue.append({
            "path": model_dir,
            "family": model_family,
            "condition": condition,
            "bit": bit_level,
            "type": m_type
        })
        
    print(f"[*] 총 {len(queue)}개의 평가 대상 모델을 발견했습니다.\n" + "="*50)
    return queue

def main():
    # 1. 평가 큐(Queue) 빌드
    eval_queue = build_evaluation_queue(MODELS_DIR)
    
    if not eval_queue:
        print("[!] 평가할 모델을 찾지 못했습니다. 경로를 확인하세요.")
        return

    # 2. CSV 저장 폴더 준비
    os.makedirs(CSV_LOG_PATH.parent, exist_ok=True)
    
    # 3. 파이프라인 가동
    for i, task in enumerate(eval_queue, 1):
        print(f"\n[ Task {i}/{len(eval_queue)} ] =======================================")
        print(f" 📂 Model : {task['family']} ({task['condition']})")
        print(f" ⚙️ Level : {task['bit']} (Type: {task['type']})")
        print(f" ⏳ Status: 로딩 및 평가 진행 중...")
        
        # 평가 함수 호출
        ppl = evaluate_model(task["path"], task["type"], DATASET_PATH, CSV_LOG_PATH)
        
        if ppl is not None:
            print(f" ✨ Result: PPL {ppl:.4f} 도출 완료!")
            
            # 실시간 로그 기록 (Dataframe append)
            df = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model_Family": task["family"],
                "Condition": task["condition"],
                "Bit_Level": task["bit"],
                "Model_Type": task["type"],
                "Perplexity": ppl
            }])
            
            file_exists = os.path.isfile(CSV_LOG_PATH)
            df.to_csv(CSV_LOG_PATH, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
            print(f" 💾 Saved : 결과가 CSV에 안전하게 기록되었습니다.")
            print(f" 🧹 VRAM  : 캐시 클리어 완료.")

    print("\n" + "="*50)
    print("🎉 모든 모델의 성능 평가 파이프라인이 성공적으로 종료되었습니다!")
    print(f"📊 최종 결과 확인: {CSV_LOG_PATH}")

if __name__ == "__main__":
    main()