import os
from datasets import load_dataset

# # 데이터셋이 저장될 폴더 생성
# save_dir = "./Evaluation_Datasets"
# os.makedirs(save_dir, exist_ok=True)

# print("1. WikiText-2 (Perplexity 평가용) 다운로드 및 추출 시작...")
# try:
#     # wikitext-2-raw-v1 버전의 테스트 데이터만 가져옵니다.
#     wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
#     # llama.cpp의 PPL 측정 도구가 읽을 수 있도록 순수 텍스트 파일로 저장합니다.
#     wiki_path = os.path.join(save_dir, "wikitext-2-test.txt")
#     with open(wiki_path, "w", encoding="utf-8") as f:
#         for item in wiki_dataset:
#             f.write(item["text"] + "\n")
#     print(f"-> 완료: {wiki_path} (텍스트 파일 생성됨)\n")
# except Exception as e:
#     print(f"WikiText 다운로드 오류: {e}\n")


# print("2. HellaSwag (상식 추론 평가용) 다운로드 시작...")
# try:
#     # HellaSwag 데이터셋 전체를 다운로드합니다.
#     hellaswag_dataset = load_dataset("Rowan/hellaswag")
    
#     # 향후 lm-evaluation-harness 등의 프레임워크에서 오프라인으로 
#     # 불러올 수 있도록 Hugging Face 로컬 디스크 포맷으로 저장합니다.
#     hellaswag_path = os.path.join(save_dir, "hellaswag")
#     hellaswag_dataset.save_to_disk(hellaswag_path)
#     print(f"-> 완료: {hellaswag_path} (로컬 데이터셋 구조로 저장됨)\n")
# except Exception as e:
#     print(f"HellaSwag 다운로드 오류: {e}\n")

# print("평가용 데이터셋 준비가 모두 완료되었습니다.")

import os
import urllib.request

def download_wikitext2_valid():
    # 1. 저장할 최상위 디렉토리 설정
    base_dir = r"C:\Users\user\SLM\03_Evaluation_Datasets"
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"디렉토리 생성 완료: {base_dir}")

    # 2. 다운로드할 파일 URL 및 저장 경로 설정
    # Wikitext-2 원본 raw 데이터 URL (Salesforce 공식 레포지토리 등에서 제공되는 형태)
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    file_path = os.path.join(base_dir, "wikitext-2-valid.txt")

    # 3. 파일 다운로드 실행
    print(f"다운로드 시작...\nURL: {url}")
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"✅ 다운로드 성공!\n저장 경로: {file_path}")
        
        # 파일 크기 확인 (정상 다운로드 검증)
        file_size = os.path.getsize(file_path)
        print(f"파일 크기: {file_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")

# 스크립트 실행
if __name__ == "__main__":
    download_wikitext2_valid()