import os
import gc
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 환경 및 경로 설정
# ==========================================
BASE_DIR = Path("/mnt/c/Users/user/SLM")
# 원본 모델들이 저장된 폴더 경로 (사용자 환경에 맞게 수정 필요)
BASE_MODELS_ROOT = BASE_DIR / "00_Base_Models" 
DATASET_PATH = BASE_DIR / "03_Evaluation_Datasets" / "wikitext-2-valid.txt"
CSV_LOG_PATH = BASE_DIR / "02_cuda_aligned" / "all_logs" / "evaluation_results_full.csv"

# ==========================================
# 2. PPL(Perplexity) 측정 함수 (동일 로직 유지)
# ==========================================
def calculate_perplexity(model, tokenizer, text_path, stride=512, max_length=2048):
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
# 3. 메인 실행 파이프라인
# ==========================================
def main():
# 평가 대상 원본 모델 리스트 (실제 폴더명 반영)
    base_tasks = [
        {"family": "Llama_3.2_1B", "path": "Llama-3.2-1B-Instruct"},
        {"family": "Qwen2.5_1.5B", "path": "Qwen2.5-1.5B-Instruct"},
        {"family": "TinyLlama_1.1B", "path": "TinyLlama-1.1B-Chat-v1.0"}
    ]
    
    for task in base_tasks:
        model_path = BASE_MODELS_ROOT / task["path"]
        print(f"\n[ Baseline ] =======================================")
        print(f" 📂 Model : {task['family']} (Original Base)")
        print(f" ⚙️ Level : 16-bit (BF16/FP16)")
        
        if not model_path.exists():
            print(f" ❌ Skip  : 경로를 찾을 수 없습니다. ({model_path})")
            continue

        try:
            # 토크나이저 및 원본 모델 로드
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), 
                device_map="auto", 
                torch_dtype=torch.bfloat16, # Llama 3.2, Qwen 2.5 권장 정밀도
                trust_remote_code=True
            )
            model.eval()

            # PPL 측정
            ppl = calculate_perplexity(model, tokenizer, DATASET_PATH)
            
            # 결과 기록
            print(f" ✨ Result: Base PPL {ppl:.4f}")
            
            df = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model_Family": task["family"],
                "Condition": "Original_Base",
                "Bit_Level": "16-bit",
                "Model_Type": "base",
                "Perplexity": ppl
            }])
            
            file_exists = os.path.isfile(CSV_LOG_PATH)
            df.to_csv(CSV_LOG_PATH, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
            
            # 메모리 정리
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            print(f" 🧹 VRAM  : 캐시 클리어 완료.")

        except Exception as e:
            print(f" ❌ Error : {e}")
            torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "="*50)
    print("🎉 모든 원본 모델의 Baseline 측정이 완료되었습니다.")

if __name__ == "__main__":
    main()