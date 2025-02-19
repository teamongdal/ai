import json
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn

def load_feature_extractor(device):
    # torchvision에서 제공하는 pretrained maskrcnn_resnet50_fpn의 backbone은 FPN의 첫 번째 레벨의 채널 수가 256입니다.
    # 이를 feature extractor로 사용합니다.
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model

def find_image(image_path):
    """이미지 파일 경로에서 PIL 이미지 객체 반환"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None

def extract_roi_feature_for_crop(cropped_image, feature_extractor, device):
    """
    주어진 crop된 이미지를 feature extractor에 통과시켜,
    maskrcnn_resnet50_fpn의 backbone으로부터 256차원 feature vector를 추출합니다.
    """
    # 입력 전처리: 간단하게 ToTensor()만 적용 (backbone은 자체 normalization을 사용하지 않으므로)
    transform = T.Compose([
        T.ToTensor()
    ])
    img_tensor = transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # backbone의 feature map은 dict 형태로 반환되며, key "0"는 256채널 feature map입니다.
        features = feature_extractor.backbone(img_tensor)
    feature_map = features["0"]  # (1, 256, H, W)
    # Adaptive average pooling하여 고정 크기 (1, 256)로 만듦
    pooled = F.adaptive_avg_pool2d(feature_map, (1, 1))
    feature_vector = pooled.view(-1).cpu().numpy()
    return feature_vector

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    input_json_path = "product_combined_cat&color.json"  # Roboflow 모델 결과가 포함된 JSON 파일
    output_npy_path = "product_features_256.npy"  # 출력 npy 파일 경로
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DF2MatchRCNN 대신 Roboflow 기반 모델을 사용한다고 가정하므로,
    # 여기서는 torchvision의 maskrcnn_resnet50_fpn의 backbone을 feature extractor로 사용합니다.
    feature_extractor = load_feature_extractor(device)
    
    json_data = load_json(input_json_path)
    all_features = []
    
    for item in json_data:
        image_path = item.get("product_images_1")
        if not image_path:
            continue
        # 만약 확장자가 없으면 .jpg 추가
        if not image_path.lower().endswith(".jpg"):
            image_path += ".jpg"
        
        image = find_image(image_path)
        if image is None:
            continue
        
        print(f"Processing image: {image_path}")
        for cloth in item.get("clothes", []):
            box = cloth.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))
            # feature vector 추출 (256차원)
            feature_vector = extract_roi_feature_for_crop(cropped, feature_extractor, device)
            all_features.append(feature_vector)
    
    np.save(output_npy_path, np.array(all_features, dtype=object))
    print(f"Extracted features saved to {output_npy_path}")
