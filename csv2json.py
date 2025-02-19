import pandas as pd
import json

# CSV 파일 경로
csv_file = "product_combined.csv"
json_file = "product_combined.json"

# CSV 파일을 데이터프레임으로 읽기
df = pd.read_csv(csv_file)

json_str = df.to_json(orient="records", force_ascii=False, indent=4)

# JSON을 직접 처리하여 슬래시 이스케이프 제거
json_str = json.dumps(json.loads(json_str), ensure_ascii=False, indent=4, separators=(",", ": "))


# JSON 파일로 저장
with open(json_file, "w", encoding="utf-8") as f:
    f.write(json_str)

print(f"JSON 파일이 저장되었습니다: {json_file}")