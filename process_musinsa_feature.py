import json
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.ops as ops

CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress",
    "long sleeve dress", "vest dress", "sling dress"
]

def extract_features_before_softmax(image_tensor, model, device):
    """ Softmax 전 단계에서 Feature Map 추출 """
    with torch.no_grad():
        outputs = model(image_tensor.to(device))

    # RoI Pooling 후의 Feature 가져오기
    boxes = outputs[0]['boxes']  # Faster R-CNN 내부에서 사용되는 RoI Features
    feature_vector = torch.flatten(boxes, start_dim=1)  # Flatten 후 벡터화

    return feature_vector

def extract_roi_features(image_tensor, model, device, detected_boxes, detected_labels):
    """ Faster R-CNN의 feature map에서 RoI Align을 이용하여 feature vector 추출 """
    with torch.no_grad():
        features = model.backbone(image_tensor.to(device))  # Backbone Feature Map 추출
    
    feature_map = features["0"]  # Feature Pyramid Networks(FPN)에서 최상위 Feature Map 사용
    batch_size, num_channels, height, width = feature_map.shape  # (1, 256, H, W)
    
    # 바운딩 박스를 PyTorch Tensor로 변환 (RoI Align은 Normalized Input이 필요함)
    boxes = torch.tensor(detected_boxes, dtype=torch.float32, device=device)  # (N, 4)
    box_indices = torch.zeros((boxes.shape[0],), dtype=torch.int32, device=device)  # 모든 RoI의 batch index는 0

    # ✅ RoI Align 적용
    aligned_features = ops.roi_align(
        feature_map,  # Feature Map
        [boxes],  # RoIs (List 형태)
        output_size=(7, 7),  # RoI Align 후 크기
        spatial_scale=1.0,  # Scaling Factor (이미 Backbone에서 줄였으므로 1.0)
        sampling_ratio=2  # 샘플링 개수 (보통 2~4 추천)
    )  # (N, C, 7, 7)

    # ✅ Feature 벡터 변환 (Flatten & Mean)
    pooled_features = aligned_features.mean(dim=[2, 3])  # (N, C)
    
    return pooled_features.cpu().numpy().tolist()

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(model_path, device):
    """ DF2MatchRCNN 모델 로드 및 Feature Extractor 설정 """
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

    # Faster R-CNN 모델 로드
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=14)
    model.load_state_dict(new_state_dict, strict=False)
    
    # Feature Extractor: backbone만 사용
    model.to(device)
    model.eval()
    return model

def download_image(image_url):
    """ 이미지 다운로드 후 PIL 객체로 변환, URL이 유효하지 않으면 None 반환 """
    if not image_url or not image_url.startswith("http"):
        return None
    try:
        response = requests.get(image_url, stream=True, timeout=5)
        if response.status_code == 200:
            return Image.open(response.raw).convert("RGB")
    except requests.exceptions.RequestException:
        return None
    return None

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
    
    print(f"Processing: {image_url}")

    # clothes = item.get("clothes", [])
    # x1, y1, x2, y2 = clothes[0]['box']
    # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
    # cropped_image = image.crop((x1, y1, x2, y2))
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        full_feature_map = extract_features_before_softmax(input_tensor, model, device)  # DF2MatchRCNN 모델 활용
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    selected_boxes, selected_labels = [], []
    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold and (label - 1) < len(CATEGORIES):
            selected_boxes.append(box)
            selected_labels.append(label)

    if not selected_boxes:
        print(f"해당 카테고리에 속하는 객체가 없습니다: {image_url}")
        return []
    
    # ✅ RoI 기반 Feature Vector 추출
    return extract_roi_features(input_tensor, model, device, selected_boxes, selected_labels)


if __name__ == "__main__":
    input_json_path = "hoodie_with_cat&color.json"  # 입력 JSON 파일 경로
    output_npy_path = "hoodie_with_feat.npy"  # 출력 .npy 파일 경로

    # 모델 로드 (DF2MatchRCNN 활용)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "df2matchrcnn"  # 모델 경로
    model = load_model(model_path, device)  # DF2MatchRCNN 모델 로드

    # JSON 데이터 로드
    json_data = load_json(input_json_path)

    # 객체 특징 추출
    all_features = []
    for item in json_data:
        features = extract_clothes(item, model, device)
        if features:  # 빈 리스트가 아닐 경우 추가
            all_features.extend(features)

    # numpy 배열로 변환 후 저장
    if all_features:
        np.save(output_npy_path, np.array(all_features, dtype=object))  # dtype=object로 설정하여 리스트 내 배열 처리
        print(f"Extracted features saved to {output_npy_path}")
    else:
        print("No features extracted.")
