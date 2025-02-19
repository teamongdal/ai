import json
import requests
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from roboflow import Roboflow

# ✅ 카테고리 리스트 (DeepFashion2 클래스)
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress",
    "trousers", "skirt", "shorts"
]

###############################################
# Roboflow 모델 로드 함수
###############################################
def load_roboflow_model(api_key):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("deepfashion2-m-11k")
    model = project.version(1).model
    return model

###############################################
# JSON 파일 로드 및 저장 함수
###############################################
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

###############################################
# 이미지 다운로드
###############################################
def download_image(image_url):
    """이미지 다운로드 후 PIL 객체로 변환"""
    if not image_url or not image_url.startswith("http"):
        return None
    try:
        response = requests.get(image_url, stream=True, timeout=5)
        if response.status_code == 200:
            return Image.open(response.raw).convert("RGB")
    except requests.exceptions.RequestException:
        return None
    return None

###############################################
# 색상 감지 (바운딩 박스 영역)
###############################################
def detect_color_kmeans_blur(image, box, k=3):
    """K-means + Blur 처리를 사용하여 바운딩 박스 내 주 색상을 감지"""
    x1, y1, x2, y2 = map(int, box)
    image_np = np.array(image)
    cropped_region = image_np[y1:y2, x1:x2]
    
    blurred = cv2.GaussianBlur(cropped_region, (5, 5), 0)
    blurred_hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    pixels = blurred_hsv.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    
    return [int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])]

###############################################
# Roboflow 모델을 이용한 객체 감지 및 색상 추출
###############################################
def extract_clothes(item, model, score_threshold=0.5):
    object_features = []
    
    image_url = item.get("product_images_1")
    if not image_url:
        print("No image URL found.")
        return []
    
    image = download_image(image_url)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_url}")
        return []
    
    # Roboflow 모델은 파일 기반 예측을 수행하므로 임시 파일에 저장
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Roboflow 예측 (confidence threshold 적용)
    predictions = model.predict(temp_path, confidence=score_threshold).json()
    
    # 임시 파일 삭제 (필요 시)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"Processing: {image_url}")
    
    for pred in predictions["predictions"]:
        # Roboflow는 [x_center, y_center, width, height] 형식으로 bbox 반환
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        score = pred["confidence"]
        category = pred["class"]
        
        # 좌표 변환: (x_center, y_center, width, height) -> (x1, y1, x2, y2)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        # 카테고리 이름 처리
        category_name = category if category in CATEGORIES else "unknown"
        
        box = [x1, y1, x2, y2]
        color_vector = detect_color_kmeans_blur(image, box)
        
        object_features.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "score": float(score),
            "category": category_name,
            "color_vector": color_vector
        })
    
    return object_features

###############################################
# JSON 데이터 처리
###############################################
def process_json_data(json_data, model, score_threshold=0.5):
    updated_data = []
    for item in json_data:
        extracted_features = extract_clothes(item, model, score_threshold)
        item["clothes"] = extracted_features
        updated_data.append(item)
        if item.get("detail_url") == "https://www.musinsa.com/products/4422105":
            break
    return updated_data

###############################################
# 실행
###############################################
if __name__ == "__main__":
    # Roboflow API 키 (자신의 API 키로 변경)
    ROBOFLOW_API_KEY = "iaKZpe4SwjpsNWkYh7aO"
    model = load_roboflow_model(ROBOFLOW_API_KEY)
    
    input_json_path = "combined.json"
    output_json_path = "combined_with_cat&color_robo.json"
    
    json_data = load_json(input_json_path)
    updated_json_data = process_json_data(json_data, model, score_threshold=0.5)
    
    filtered_data = [item for item in updated_json_data if "clothes" in item and isinstance(item["clothes"], list) and len(item["clothes"]) > 0]
    
    save_json(filtered_data, output_json_path)
    print(f"Processed JSON saved to {output_json_path}")
