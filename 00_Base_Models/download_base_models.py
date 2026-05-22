import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# ==========================================
# [설정 1] 다운로드 최상위 경로 (Windows 규격 적용)
# ==========================================
# 백슬래시(\) 대신 슬래시(/)를 사용하여 경로 인식 오류를 원천 차단합니다.
BASE_DIR = "C:/Users/user/SLM/00_Base_Models"

# ==========================================
# [설정 2] 모델 리스트 및 식별 폴더 매핑
# ==========================================
# 딕셔너리 구조: {"HuggingFace_Repo_ID": "로컬_저장_폴더명"}
models_to_download = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct"
}

# ==========================================
# [설정 3] Hugging Face Access Token 
# ==========================================
# Llama-3.2 모델을 다운로드하기 위한 필수 인증 키입니다.
# Hugging Face 홈페이지(Settings -> Access Tokens)에서 발급받은 'Read' 권한의 토큰을 입력하십시오.
# .env 파일 읽어오기 (pip install python-dotenv 필요)
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# ==========================================
# [실행] 순차적 다운로드 파이프라인
# ==========================================
print("=" * 60)
print("[원본 모델 다운로드 파이프라인 시작]")
print("=" * 60)

for repo_id, folder_name in models_to_download.items():
    save_path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n-> [{repo_id}] 원본 다운로드 준비 중...")
    print(f"-> 대상 경로: {save_path}")

    try:
        # snapshot_download: 리포지토리의 원본 상태를 그대로 복제
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_path,
            local_dir_use_symlinks=False, # 심볼릭 링크 배제, 실제 파일 물리적 저장 강제
            token=HF_TOKEN
        )
        print(f"-> [성공] {folder_name} 모델의 순수 원본 확보 완료.")
        
    except Exception as e:
        print(f"-> [실패] {folder_name} 모델 다운로드 중 에러 발생:")
        print(f"Error Message: {e}")

print("\n" + "=" * 60)
print("지정된 3종 모델에 대한 모든 다운로드 프로세스가 종료되었습니다.")