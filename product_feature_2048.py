import json
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50

def load_feature_extractor(device):
    # ImageNet pretrained ResNet-50에서 마지막 fc layer를 제거하여 feature extractor로 사용
    model = resnet50(pretrained=True)
    model.fc = nn.Identity()
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

def extract_feature_for_cloth(image, box, feature_extractor, device):
    """
    주어진 box 좌표([x1, y1, x2, y2])에 해당하는 영역을 crop한 후,
    ResNet-50 feature extractor를 이용해 feature vector를 추출합니다.
    """
    x1, y1, x2, y2 = map(int, box)
    cropped = image.crop((x1, y1, x2, y2))
    
    # 전처리: ResNet 입력 크기 224x224, ImageNet 정규화 적용
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    cropped_tensor = transform(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = feature_extractor(cropped_tensor)
    # feature vector를 numpy 배열로 반환 (예: 2048 차원)
    return feature.squeeze().cpu().numpy()

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    input_json_path = "product_combined_cat&color.json"  # 입력 JSON 파일 경로
    output_npy_path = "product_features_2048.npy"  # 출력 npy 파일 경로
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = load_feature_extractor(device)
    
    json_data = load_json(input_json_path)
    all_features = []
    
    for item in json_data:
        image_path = item.get("product_images_1")
        if not image_path:
            continue
        # 확장자가 없으면 .jpg 추가
        if not image_path.lower().endswith(".jpg"):
            image_path += ".jpg"
        
        image = find_image(image_path)
        if image is None:
            continue
        
        print(f"Processing image: {image_path}")
        # 각 cloth 객체의 box 좌표에 대해 feature vector 추출
        for cloth in item.get("clothes", []):
            box = cloth.get("box")
            if not box or len(box) != 4:
                continue
            feature_vector = extract_feature_for_cloth(image, box, feature_extractor, device)
            all_features.append(feature_vector)
    
    # numpy 배열로 변환 후 npy 파일로 저장
    np.save(output_npy_path, np.array(all_features))
    print(f"Extracted features saved to {output_npy_path}")
