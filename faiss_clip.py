import faiss
import numpy as np
import json
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.models import resnet50
from PIL import Image
import cv2
import os
from roboflow import Roboflow
from sklearn.cluster import KMeans

# Fashion‑CLIP 관련 임포트 (카테고리 분류용)
from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor

# 표준 CLIP 모델을 style attribute 추출용으로 로드 (scripy.py 방식)
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
style_model, style_preprocess = clip.load("ViT-B/32", device=device)

# 텍스트 임베딩 함수 (정규화 포함)
def encode_texts(texts):
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = style_model.encode_text(text_tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# 텍스트 벡터 (각 속성별)
TEXTURE_CATEGORIES = ["rough", "soft", "glossy"]
SEASON_CATEGORIES = ["spring", "summer", "fall", "winter"]
MOOD_CATEGORIES = ["casual", "elegant", "sporty"]

texture_vectors = encode_texts(TEXTURE_CATEGORIES).cpu().numpy()  # shape: (3, 512)
season_vectors  = encode_texts(SEASON_CATEGORIES).cpu().numpy()   # shape: (4, 512)
mood_vectors    = encode_texts(MOOD_CATEGORIES).cpu().numpy()       # shape: (3, 512)

# 이미지의 스타일 임베딩을 추출하는 함수 (scripy.py 방식)
def classify_feature_vector(image, text_vectors):
    image_input = style_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = style_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ torch.tensor(text_vectors, device=device).T)
    return similarity.cpu().numpy().flatten()  # 예: 3, 4, 3 차원 벡터

# --- 설정 ---
# PRODUCT_JSON는 추가된 texture/season/mood vector가 있으므로 수정
FEATURES_NPY = "product_features_2048.npy"  # 2048-dim feature vector
PRODUCT_JSON = "product_combined_cat&color&clip.json"  # JSON 파일에 texture_vector, season_vector, mood_vector 포함
ROBOFLOW_API_KEY = "iaKZpe4SwjpsNWkYh7aO"  # Roboflow API 키
QUERY_IMAGE_PATH = "A1_1.jpg"  # 예시 query 이미지 파일 경로
K = 5  # 추천 상위 k개

# 기존의 카테고리 리스트(이제 combined similarity에서 사용하지 않음 – 대신 색상, 텍스처, 시즌, 무드를 사용)
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress",
    "trousers", "skirt", "shorts"
]

# --- 기존 함수들 (faiss_color_cat.py) ---
def load_fashion_clip_model():
    # Fashion‑CLIP 모델을 불러와서 fclip.processor도 저장 (카테고리 분류용)
    fclip = FashionCLIP("fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    fclip.processor = processor
    return fclip

def load_resnet_feature_extractor(device):
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

def load_roboflow_model(api_key):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("deepfashion2-m-11k")
    model = project.version(1).model
    return model

def find_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_feature_for_crop_resnet(cropped_image, feature_extractor, device):
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = feature_extractor(img_tensor)
    return feature.squeeze().cpu().numpy()

# Roboflow 기반 객체 검출 (box와 초기 카테고리 추출)
def detect_query_box(query_image, roboflow_model, score_threshold=0.5):
    temp_path = "temp_query.jpg"
    query_image.save(temp_path)
    predictions = roboflow_model.predict(temp_path, confidence=score_threshold).json()
    os.remove(temp_path)
    if "predictions" not in predictions or len(predictions["predictions"]) == 0:
        width, height = query_image.size
        return [0, 0, width, height], None
    preds = predictions["predictions"]
    preds.sort(key=lambda x: x["confidence"], reverse=True)
    best = preds[0]
    x, y, w, h = best["x"], best["y"], best["width"], best["height"]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return [x1, y1, x2, y2], best.get("class", None)

def detect_query_category_clip(query_image, query_box, fclip, categories):
    cropped = query_image.crop(tuple(map(int, query_box)))
    image_tensor = fclip.processor(images=cropped, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        image_feature = fclip.model.get_image_features(image_tensor)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    text_inputs = fclip.processor(text=categories, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = fclip.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarities = (image_feature @ text_features.T).squeeze(0)
    best_index = similarities.argmax().item()
    return categories[best_index]

def detect_color_kmeans_blur(image, box, k=3):
    x1, y1, x2, y2 = map(int, box)
    image_np = np.array(image)
    cropped_region = image_np[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(cropped_region, (5,5), 0)
    blurred_hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    pixels = blurred_hsv.reshape((-1,3))
    kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return [int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])]

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def map_index_to_json(idx, products):
    count = 0
    for prod_idx, product in enumerate(products):
        clothes = product.get("clothes", [])
        for box_idx, cloth in enumerate(clothes):
            if count == idx:
                return prod_idx, box_idx
            count += 1
    return None, None

# --- Main processing ---
if __name__ == "__main__":
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Fashion‑CLIP 모델 (카테고리 분류용) 로드
    fclip = load_fashion_clip_model()
    
    # ResNet50 feature extractor 로드
    resnet_feature_extractor = load_resnet_feature_extractor(device)
    
    # Roboflow 모델 로드
    roboflow_model = load_roboflow_model(ROBOFLOW_API_KEY)
    
    # 제품 feature 벡터 로드 (np.load)
    product_features = np.load(FEATURES_NPY, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(product_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    product_features /= norms
    print("Product features loaded. Shape:", product_features.shape)
    
    # 제품 JSON 로드 (추가된 texture_vector, season_vector, mood_vector 포함)
    products = load_json(PRODUCT_JSON)
    
    # Query 이미지 로드
    query_image = Image.open(QUERY_IMAGE_PATH).convert("RGB")
    
    # Roboflow를 이용한 query 영역 및 초기 카테고리 검출
    query_box, query_category = detect_query_box(query_image, roboflow_model, score_threshold=0.5)
    print("Initial Query box:", query_box, "Initial Query category from Roboflow:", query_category)
    
    # (카테고리 분류는 기존 방식대로 진행 – 필요 시 Fashion‑CLIP 기반 분류 사용)
    if query_category is None or query_category.strip().lower() not in [c.lower() for c in CATEGORIES]:
        clip_category = detect_query_category_clip(query_image, query_box, fclip, CATEGORIES)
        print("Fashion‑CLIP detected category:", clip_category)
        query_category = clip_category
    else:
        clip_category = detect_query_category_clip(query_image, query_box, fclip, CATEGORIES)
        if clip_category.strip().lower() != query_category.strip().lower():
            print("Mismatch between Roboflow and Fashion‑CLIP categories. Using Fashion‑CLIP's result.")
            query_category = clip_category
    print("Final Query category:", query_category)
    
    # Query crop 및 2048-dim feature vector 추출 (ResNet50 기반)
    q_x1, q_y1, q_x2, q_y2 = map(int, query_box)
    cropped_query = query_image.crop((q_x1, q_y1, q_x2, q_y2))
    query_feature = extract_feature_for_crop_resnet(cropped_query, resnet_feature_extractor, device)
    query_feature = query_feature.astype(np.float32)
    q_norm = np.linalg.norm(query_feature)
    if q_norm == 0:
        q_norm = 1
    query_feature /= q_norm
    query_feature = query_feature.reshape(1, -1)
    
    # Query 색상 벡터 추출
    query_color = detect_color_kmeans_blur(query_image, query_box)
    query_color = np.array(query_color, dtype=float)
    norm_color = np.linalg.norm(query_color)
    if norm_color == 0:
        norm_color = 1
    query_color /= norm_color
    
    # Query 스타일 속성 벡터 추출 (표준 CLIP 이용, scripy.py 방식)
    query_texture = classify_feature_vector(cropped_query, texture_vectors)  # 3-dim
    query_season  = classify_feature_vector(cropped_query, season_vectors)   # 4-dim
    query_mood    = classify_feature_vector(cropped_query, mood_vectors)     # 3-dim
    
    # Faiss 인덱스 생성 (2048-dim feature vector 기반)
    d = product_features.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, 5, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        index.train(product_features)
    index.add(product_features)
    
    D, I = index.search(query_feature, K)
    print("추천 인덱스 (npy):", I[0])
    
    # Combined similarity 계산
    # 가중치 설정 (필요에 따라 조정)
    alpha = 0.5   # feature similarity weight
    beta  = 0.2   # color similarity weight
    gamma = 0.1   # texture similarity weight
    delta = 0.1   # season similarity weight
    epsilon = 0.1 # mood similarity weight
    
    combined_scores = []
    for i in range(product_features.shape[0]):
        feat_sim = cosine_similarity(query_feature.flatten(), product_features[i])
        prod_idx, box_idx = map_index_to_json(i, products)
        if prod_idx is None:
            color_sim = 0
            texture_sim = 0
            season_sim = 0
            mood_sim = 0
        else:
            prod_cloth = products[prod_idx]["clothes"][box_idx]
            # 색상 similarity
            prod_color = prod_cloth.get("color_vector")
            if prod_color is None:
                color_sim = 0
            else:
                prod_color = np.array(prod_color, dtype=float)
                norm_prod = np.linalg.norm(prod_color)
                if norm_prod == 0:
                    norm_prod = 1
                prod_color /= norm_prod
                color_sim = cosine_similarity(query_color, prod_color)
            # 텍스처 similarity
            prod_texture = prod_cloth.get("texture_vector")
            if prod_texture is None:
                texture_sim = 0
            else:
                prod_texture = np.array(prod_texture, dtype=float)
                norm_prod = np.linalg.norm(prod_texture)
                if norm_prod == 0:
                    norm_prod = 1
                prod_texture /= norm_prod
                texture_sim = cosine_similarity(query_texture, prod_texture)
            # 시즌 similarity
            prod_season = prod_cloth.get("season_vector")
            if prod_season is None:
                season_sim = 0
            else:
                prod_season = np.array(prod_season, dtype=float)
                norm_prod = np.linalg.norm(prod_season)
                if norm_prod == 0:
                    norm_prod = 1
                prod_season /= norm_prod
                season_sim = cosine_similarity(query_season, prod_season)
            # 무드 similarity
            prod_mood = prod_cloth.get("mood_vector")
            if prod_mood is None:
                mood_sim = 0
            else:
                prod_mood = np.array(prod_mood, dtype=float)
                norm_prod = np.linalg.norm(prod_mood)
                if norm_prod == 0:
                    norm_prod = 1
                prod_mood /= norm_prod
                mood_sim = cosine_similarity(query_mood, prod_mood)
        combined = (alpha * feat_sim +
                    beta * color_sim +
                    gamma * texture_sim +
                    delta * season_sim +
                    epsilon * mood_sim)
        combined_scores.append(combined)
    combined_scores = np.array(combined_scores)
    sorted_indices = np.argsort(-combined_scores)
    unique_indices = []
    for idx in sorted_indices:
        if idx not in unique_indices:
            unique_indices.append(idx)
        if len(unique_indices) >= K:
            break
    print("추천 인덱스 (combined):", unique_indices)
    print("유사도 점수 (combined):", combined_scores[unique_indices])
    
    mapping = []
    for npy_idx in unique_indices:
        prod_idx, box_idx = map_index_to_json(npy_idx, products)
        if prod_idx is not None:
            mapping.append((npy_idx, prod_idx, box_idx))
    print("매핑 결과 (npy index, product index, box index):", mapping)
    
    for npy_idx, prod_idx, box_idx in mapping:
        product = products[prod_idx]
        img_path = product.get("product_images_1")
        if not img_path:
            continue
        if not img_path.lower().endswith(".jpg"):
            img_path += ".jpg"
        prod_img = cv2.imread(img_path)
        if prod_img is None:
            continue
        cloth = product.get("clothes", [])[box_idx]
        box = cloth.get("box")
        score = cloth.get("score", 0)
        prod_cat = cloth.get("category", "unknown")
        prod_name = product.get("product_code", "NoName")
        if box and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(prod_img, (x1, y1), (x2, y2), (0,255,0), 2)
            text = f"{prod_name} | {score:.2f} | {prod_cat}"
            cv2.putText(prod_img, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            color_vector = cloth.get("color_vector")
            if color_vector is not None:
                hsv_color = np.uint8([[color_vector]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                patch_w, patch_h = 40, 40
                patch_x1 = x2 - patch_w
                patch_y1 = y1
                patch_x2 = x2
                patch_y2 = y1 + patch_h
                cv2.rectangle(prod_img, (patch_x1, patch_y1), (patch_x2, patch_y2), bgr_color.tolist(), -1)
                color_text = f"HSV: {color_vector}"
                cv2.putText(prod_img, color_text, (patch_x1, patch_y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        window_name = f"Product {prod_idx} Box {box_idx}"
        cv2.imshow(window_name, prod_img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()
