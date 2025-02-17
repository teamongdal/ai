import json
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# JSON 파일 로드 함수
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 이미지 다운로드 함수
def download_image(image_url):
    try:
        response = requests.get(image_url, stream=True, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return np.array(image)
    except requests.exceptions.RequestException:
        return None
    return None

def visualize_product(json_data, url):
    for item in json_data:
        if item["detail_url"] == url:
            image_url = item["product_images_1"]  # 첫 번째 이미지 사용
            clothes = item.get("clothes", [])

            # 이미지 다운로드
            image = download_image(image_url)
            if image is None:
                print("이미지를 불러올 수 없습니다.")
                return

            # OpenCV에서 사용할 BGR 이미지 변환
            image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # JSON에서 "box" 정보 가져와서 Bounding Box 그리기
            for obj in clothes:
                box = obj["box"]  # Bounding Box 좌표
                category = obj["category"]  # 카테고리명
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_cv2, category, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 이미지 출력
            cv2.imshow(f"Product Code: {url}", image_cv2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
    
    print("해당 Product Code를 찾을 수 없습니다.")

if __name__ == "__main__":
    json_file_path = "hoodie_with_cat&color.json"
    json_data = load_json(json_file_path)

    url = input("url를 입력하세요: ")
    visualize_product(json_data, url)
