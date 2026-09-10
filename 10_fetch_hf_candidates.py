from huggingface_hub import HfApi
import csv

api = HfApi()

models = list(api.list_models(
    pipeline_tag="text-generation",
    language="en",
    filter=["safetensors", "transformers", "license:apache-2.0"],
    sort="downloads",
    direction=-1,
    limit=500,
    full=True,       # safetensors.total(파라미터 수) 등 추가 정보 포함
    cardData=True,
))

def is_derived(tags):
    prefixes = ("base_model:quantized:", "base_model:finetune:",
                "base_model:adapter:", "base_model:merge:")
    return any(t.startswith(prefixes) for t in (tags or []))

rows = []
for m in models:
    param_count = None
    if getattr(m, "safetensors", None) and getattr(m.safetensors, "total", None):
        param_count = m.safetensors.total
    rows.append([m.id, m.downloads, m.likes, param_count,
                 is_derived(m.tags), ",".join(m.tags or [])])

with open("candidates_raw.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "downloads", "likes", "param_count", "is_derived", "tags"])
    writer.writerows(rows)

print(f"{len(rows)}개 저장 완료 -> candidates_raw.csv")