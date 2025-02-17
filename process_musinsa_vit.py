import json
import requests
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # ✅ Faster R-CNN 사용
import timm  # ✅ Vision Transformer 라이브러리 추가

# ✅ 카테고리 리스트 (DeepFashion2 클래스)
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress",
    "long sleeve dress", "vest dress", "sling dress"
]

# ✅ Faster R-CNN 모델 로드
def load_faster_rcnn(model_path, device):
    """ Faster R-CNN 모델 로드 """
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=14)  # ✅ Faster R-CNN 사용
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

# ✅ Vision Transformer(ViT) 모델 로드
def load_vit_classifier(device):
    """ 사전 학습된 ViT 모델 로드 및 분류기 설정 """
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=len(CATEGORIES))
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

    blurred = cv2.GaussianBlur(cropped_region, (5, 5), 0)
    blurred_hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    pixels = blurred_hsv.reshape((-1, 3))

    kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    return [int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])]

# ✅ ViT 기반 세부 카테고리 분류
def classify_with_vit(image, box, vit_model, device):
    """ Vision Transformer(ViT)를 사용하여 바운딩 박스 내 객체의 카테고리를 분류 """
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image.crop((x1, y1, x2, y2))  # ✅ 바운딩 박스 크롭

    # ViT 입력 변환
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = transform(cropped_image).unsqueeze(0).to(device)

    # ViT 예측
    with torch.no_grad():
        outputs = vit_model(input_tensor)
        predicted_label = torch.argmax(outputs, dim=1).item()

    return CATEGORIES[predicted_label]

# ✅ Faster R-CNN + ViT 기반 객체 탐지 및 분류
def extract_clothes(item, faster_rcnn, vit_model, device, score_threshold=0.5):
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
        outputs = faster_rcnn(input_tensor)

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    print(f"Processing: {image_url}")

    for box, score in zip(boxes, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = map(int, box)

            # ✅ Vision Transformer 기반 카테고리 분류
            category_name = classify_with_vit(image, box, vit_model, device)

            # ✅ 바운딩 박스 기반 색상 감지
            color_vector = detect_color_kmeans_blur(image, box)

            object_features.append({
                "box": [x1, y1, x2, y2],
                "category": category_name,  # ✅ ViT 예측 결과 적용
                "color_vector": color_vector
            })

    return object_features

# ✅ JSON 데이터 처리
def process_json_data(json_data, faster_rcnn, vit_model, device):
    updated_data = []
    for item in json_data:
        extracted_features = extract_clothes(item, faster_rcnn, vit_model, device)
        item["clothes"] = extracted_features
        updated_data.append(item)
    return updated_data

# ✅ 실행
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    faster_rcnn = load_faster_rcnn("df2matchrcnn", device)
    vit_model = load_vit_classifier(device)

    json_data = load_json("hoodie.json")
    updated_json_data = process_json_data(json_data, faster_rcnn, vit_model, device)

    filtered_data = [item for item in updated_json_data if not ("clothes" in item and isinstance(item["clothes"], list) and len(item["clothes"]) > 1)]

    save_json(filtered_data, "hoodie_with_vit.json")
    print("Processed JSON saved with ViT-enhanced classification!")
