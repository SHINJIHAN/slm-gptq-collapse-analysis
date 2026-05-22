import os
import gc
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel
import lm_eval
from lm_eval.models.huggingface import HFLM

# ---------------------------------------------------------
# 1. 환경 및 기본 경로 설정
# ---------------------------------------------------------
os.environ["HF_DATASETS_CACHE"] = "/mnt/c/Users/user/SLM/03_Evaluation_Datasets/cache"

BASE_MODELS_DIR = "/mnt/c/Users/user/SLM/00_Base_Models"
QUANT_MODELS_DIR = "/mnt/c/Users/user/SLM/02_cuda_aligned"
OUTPUT_DIR = "/mnt/c/Users/user/SLM/03_Evaluation_Datasets/results_cognitive"

# 결과 저장 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. 모델 및 경로 매핑 구조체 (사진 기반 정확한 경로 매핑)
# ---------------------------------------------------------
models_config = {
    "Llama-3.2-1B": {
        "base_path": os.path.join(BASE_MODELS_DIR, "Llama-3.2-1B-Instruct"),
        "quant_dir": os.path.join(QUANT_MODELS_DIR, "Llama_3.2_1B", "Distillation")
    },
    "Qwen2.5-1.5B": {
        "base_path": os.path.join(BASE_MODELS_DIR, "Qwen2.5-1.5B-Instruct"),
        "quant_dir": os.path.join(QUANT_MODELS_DIR, "Qwen2.5_1.5B", "Base_RLHF")
    },
    "TinyLlama-1.1B": {
        "base_path": os.path.join(BASE_MODELS_DIR, "TinyLlama-1.1B-Chat-v1.0"),
        "quant_dir": os.path.join(QUANT_MODELS_DIR, "TinyLlama_1.1B", "Base_Scratch")
    }
}

# 평가할 비트 목록 (원본 포함)
bit_targets = ["Base_16bit", "GPTQ_8bit", "GPTQ_4bit", "GPTQ_3bit", "GPTQ_2bit"]
eval_tasks = ["hellaswag", "piqa"]

# ---------------------------------------------------------
# 3. 평가 파이프라인 실행
# ---------------------------------------------------------
print("==================================================")
print("🚀 본격적인 인지적 붕괴 측정 (HellaSwag & PIQA) 파이프라인 시작")
print("==================================================")

for model_name, paths in models_config.items():
    for bit in bit_targets:
        print(f"\n[{model_name} | {bit}] 평가를 준비합니다...")
        
        # 결과 파일 경로 지정
        result_file_path = os.path.join(OUTPUT_DIR, f"{model_name}_{bit}_results.json")
        
        # 이미 평가된 결과가 있다면 스킵 (중간에 끊겼을 때 이어하기 용도)
        if os.path.exists(result_file_path):
            print(f"✅ 이미 완료된 평가입니다. 건너뜁니다: {result_file_path}")
            continue
            
        try:
            # ---------------------------------------------------------
            # [단계 A] VRAM 적재 (Base vs Quant 분기 처리)
            # ---------------------------------------------------------
            if bit == "Base_16bit":
                current_model_path = paths["base_path"]
                print(f"-> Base 모델 로드 중 (bfloat16): {current_model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    current_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:0"
                )
                tokenizer = AutoTokenizer.from_pretrained(current_model_path)
            else:
                current_model_path = os.path.join(paths["quant_dir"], bit)
                print(f"-> 양자화 모델 로드 중 (GPTQModel): {current_model_path}")
                model = GPTQModel.from_quantized(current_model_path, device="cuda:0")
                tokenizer = AutoTokenizer.from_pretrained(current_model_path)

            # ---------------------------------------------------------
            # [단계 B] lm-eval 래핑 및 평가 실행
            # ---------------------------------------------------------
            print("-> lm_eval HFLM 래핑 및 문제 풀이 시작 (배치사이즈=1, Zero-shot)")
            lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
            
            results = lm_eval.simple_evaluate(
                model=lm_eval_model,
                tasks=eval_tasks,
                num_fewshot=0,
                # limit=10 옵션은 본 평가이므로 완전히 제거됨
            )
            
            # ---------------------------------------------------------
            # [단계 C] 결과 저장
            # ---------------------------------------------------------
            with open(result_file_path, "w", encoding="utf-8") as f:
                json.dump(results['results'], f, indent=4, ensure_ascii=False)
            print(f"✅ [{model_name} | {bit}] 평가 완료 및 저장 성공!")

        except Exception as e:
            print(f"❌ [{model_name} | {bit}] 평가 중 오류 발생: {e}")

        finally:
            # ---------------------------------------------------------
            # [단계 D] OOM 방지를 위한 강력한 메모리 회수 로직 (Zero-Impact)
            # ---------------------------------------------------------
            print("-> VRAM 정리 및 다음 모델을 위한 메모리 회수 진행...")
            
            # 1. 변수 참조 해제
            if 'results' in locals(): del results
            if 'lm_eval_model' in locals(): del lm_eval_model
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            
            # 2. 강제 가비지 컬렉션 및 CUDA 캐시 초기화
            gc.collect()
            torch.cuda.empty_cache()
            
            # 3. OS 및 GPU가 메모리를 완전히 반환할 수 있도록 5초 대기
            time.sleep(5)

print("\n🎉 모든 모델과 비트에 대한 HellaSwag & PIQA 평가가 완벽하게 종료되었습니다!")