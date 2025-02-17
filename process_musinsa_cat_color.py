import json
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
import numpy as np
import cv2
from sklearn.cluster import KMeans
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # ✅ Faster R-CNN으로 변경

# ✅ 카테고리 리스트 (DeepFashion2 클래스)
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress",
    "long sleeve dress", "vest dress", "sling dress", "shorts", "trousers", "skirt"
]

# ✅ Faster R-CNN 모델 로드
def load_model(model_path, device):
    """ DF2MatchRCNN 모델 로드 및 Feature Extractor 설정 """
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=14)  # ✅ Faster R-CNN 사용
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

# ✅ JSON 파일 로드 및 저장 함수
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ✅ 이미지 다운로드
def download_image(image_url):
    """ 이미지 다운로드 후 PIL 객체로 변환 """
    if not image_url or not image_url.startswith("http"):
        return None
    try:
        response = requests.get(image_url, stream=True, timeout=5)
        if response.status_code == 200:
            return Image.open(response.raw).convert("RGB")
    except requests.exceptions.RequestException:
        return None
    return None

# ✅ Faster R-CNN 기반 색상 감지 (마스크 없이 바운딩 박스만 사용)
def detect_color_kmeans_blur(image, box, k=3):
    """ K-means + Blur 처리를 사용하여 바운딩 박스 내에서 주 색상을 감지 """
    x1, y1, x2, y2 = map(int, box)
    image_np = np.array(image)
    cropped_region = image_np[y1:y2, x1:x2]  # ✅ 바운딩 박스 영역 크롭

    # ✅ 마스크 제거, 단순히 바운딩 박스 영역을 기반으로 색상 감지
    blurred = cv2.GaussianBlur(cropped_region, (5, 5), 0)
    blurred_hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    pixels = blurred_hsv.reshape((-1, 3))

    kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    return [int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])]

# ✅ Faster R-CNN을 이용한 객체 감지
def extract_clothes(item, model, device, score_threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    object_features = []

    image_url = item.get("product_images_1")
    if not image_url:
        print("No image URL found.")
        return []

    image = download_image(image_url)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_url}")
        return []
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    print(f"Processing: {image_url}")

    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            x1, y1, x2, y2 = map(int, box)
            category_idx = label - 1
            category_name = CATEGORIES[category_idx] if 0 <= category_idx < len(CATEGORIES) else "unknown"

            # ✅ 바운딩 박스 기반 색상 감지 (마스크 제거)
            color_vector = detect_color_kmeans_blur(image, box)

            object_features.append({
                "box": [x1, y1, x2, y2],
                "category": category_name,
                "color_vector": color_vector
            })

    return object_features

# ✅ JSON 데이터 처리
def process_json_data(json_data, model, device):
    updated_data = []
    for item in json_data:
        extracted_features = extract_clothes(item, model, device)  # ✅ 중복 호출 방지
        item["clothes"] = extracted_features
        updated_data.append(item)
    return updated_data

# ✅ 실행
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "df2matchrcnn"
    model = load_model(model_path, device)

    input_json_path = "hoodie.json"
    output_json_path = "hoodie_with_cat&color.json"

    json_data = load_json(input_json_path)
    updated_json_data = process_json_data(json_data, model, device)

    filtered_data = [item for item in updated_json_data if not ("clothes" in item and isinstance(item["clothes"], list) and len(item["clothes"]) > 1)]
    filtered_data = [item for item in filtered_data if "clothes" in item and isinstance(item["clothes"], list) and len(item["clothes"]) > 0]

    save_json(filtered_data, output_json_path)
    print(f"Processed JSON saved to {output_json_path}")
