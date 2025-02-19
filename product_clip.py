import json
import torch
from PIL import Image
import os
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Fashion‑CLIP 모델 로드 (Fashion‑CLIP과 CLIPProcessor를 함께 사용)
fclip = FashionCLIP("fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
fclip.processor = processor  # fclip 객체에 processor 저장

# 속성 리스트
TEXTURE_CATEGORIES = ["rough", "soft", "glossy"]
SEASON_CATEGORIES = ["spring", "summer", "fall", "winter"]
MOOD_CATEGORIES = ["casual", "elegant", "sporty"]

# JSON 파일 로드
json_path = "product_combined_cat&color.json"
with open(json_path, "r", encoding="utf-8") as f:
    products = json.load(f)

# 텍스트 임베딩 함수 (정규화 포함)
def encode_texts(texts):
    text_inputs = fclip.processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = fclip.model.get_text_features(**text_inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# 각 텍스트 카테고리(속성)의 벡터를 numpy 배열로 반환
def get_text_vectors(categories):
    text_features = encode_texts(categories)
    return text_features.cpu().numpy()

# PIL 이미지에서 Fashion‑CLIP 이미지 임베딩 추출 함수
def encode_image_from_pil(image):
    image_inputs = fclip.processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = fclip.model.get_image_features(image_inputs["pixel_values"])
    return image_features / image_features.norm(dim=-1, keepdim=True)

# 이미지와 텍스트 벡터 간의 cosine similarity를 계산하여 해당 이미지의 속성 벡터를 얻음
def classify_feature_vector(image, text_vectors):
    image_feature = encode_image_from_pil(image)  # (1, D)
    similarity = (image_feature @ torch.tensor(text_vectors, device=device).T)
    return similarity.cpu().numpy().flatten()

# 각 속성(텍스처, 시즌, 무드)의 임베딩 벡터 생성
texture_vectors = get_text_vectors(TEXTURE_CATEGORIES)  # (3, D)
season_vectors = get_text_vectors(SEASON_CATEGORIES)      # (4, D)
mood_vectors = get_text_vectors(MOOD_CATEGORIES)          # (3, D)

# 제품별로 clothes 항목 내에서 속성 벡터 계산
for product in products:
    print(f"Processing: {product['product_images_1']}")
    if "product_images_1" in product and product["product_images_1"] != "없음":
        image_path = product["product_images_1"] + ".jpg"
        if os.path.exists(image_path):
            original_image = Image.open(image_path).convert("RGB")
            for cloth in product.get("clothes", []):
                if "box" in cloth:
                    x1, y1, x2, y2 = map(int, cloth["box"])
                    cropped_img = original_image.crop((x1, y1, x2, y2))
                    cloth["texture_vector"] = classify_feature_vector(cropped_img, texture_vectors).tolist()
                    cloth["season_vector"] = classify_feature_vector(cropped_img, season_vectors).tolist()
                    cloth["mood_vector"] = classify_feature_vector(cropped_img, mood_vectors).tolist()

# 수정된 JSON 저장
updated_json_path = "product_combined_cat&color&clip.json"
with open(updated_json_path, "w", encoding="utf-8") as f:
    json.dump(products, f, indent=4, ensure_ascii=False)

print(f"Updated JSON saved at: {updated_json_path}")
