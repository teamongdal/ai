import faiss
import numpy as np
import json
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models import resnet50
from PIL import Image
import cv2
import requests
import os
from roboflow import Roboflow
from sklearn.cluster import KMeans

# --- 설정 ---
FEATURES_NPY = "product_features_2048.npy"  # 2048-dim feature vector들이 저장된 npy 파일
PRODUCT_JSON = "product_combined_cat&color.json"  # 제품 JSON 파일 (clothes 항목에 box, color_vector, category 포함)
ROBOFLOW_API_KEY = "iaKZpe4SwjpsNWkYh7aO"  # Roboflow API 키 (수정)
QUERY_IMAGE_PATH = "tear_queen2.jpg"  # query 이미지 파일 경로
K = 5  # 추천 상위 k개

# --- 카테고리 리스트 ---
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress",
    "trousers", "skirt", "shorts"
]

# --- 모델 및 로드 함수 ---
def load_resnet_feature_extractor(device):
    # ResNet50 pretrained 모델에서 fc를 Identity로 교체하면 출력이 2048차원입니다.
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

# --- Feature extraction for a crop using ResNet50 (2048-dim) ---
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

# --- Roboflow 기반 query detection ---
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

# --- 색상 추출 (K-means + GaussianBlur) ---
def detect_color_kmeans_blur(image, box, k=3):
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

# --- Cosine similarity ---
def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

# --- Mapping function: npy index -> JSON product & box index ---
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 로드: product_features_2048.npy (각 배열은 2048-dim feature vector)
    product_features = np.load(FEATURES_NPY, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(product_features, axis=1, keepdims=True)
    norms[norms==0] = 1
    product_features /= norms
    print("Product features loaded. Shape:", product_features.shape)
    
    # JSON 데이터 로드
    products = load_json(PRODUCT_JSON)
    
    # 로드: Roboflow 모델 (검출용)
    roboflow_model = load_roboflow_model(ROBOFLOW_API_KEY)
    
    # 로드: ResNet50 feature extractor (2048-dim)
    resnet_feature_extractor = load_resnet_feature_extractor(device)
    
    # Query 이미지 로드
    query_image = Image.open(QUERY_IMAGE_PATH).convert("RGB")
    
    # Roboflow를 이용해 query 이미지에서 객체 검출 -> box 및 카테고리 추출
    query_box, query_category = detect_query_box(query_image, roboflow_model, score_threshold=0.5)
    print("Query box:", query_box, "Query category:", query_category)
    
    # Crop query image & extract 2048-dim feature vector using ResNet50
    q_x1, q_y1, q_x2, q_y2 = map(int, query_box)
    cropped_query = query_image.crop((q_x1, q_y1, q_x2, q_y2))
    query_feature = extract_feature_for_crop_resnet = extract_feature_for_crop_resnet = extract_feature_for_crop_resnet(cropped_query, resnet_feature_extractor, device)
    query_feature = query_feature.astype(np.float32)
    q_norm = np.linalg.norm(query_feature)
    if q_norm == 0: q_norm = 1
    query_feature /= q_norm
    query_feature = query_feature.reshape(1, -1)
    
    # Query 색상 추출
    query_color = detect_color_kmeans_blur(query_image, query_box)
    query_color = np.array(query_color, dtype=float)
    norm_color = np.linalg.norm(query_color)
    if norm_color == 0: norm_color = 1
    query_color /= norm_color
    
    # --- Faiss 인덱스 생성 (2048-dim feature vector 기반) ---
    d = product_features.shape[1]  # d should be 2048 now
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, 5, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        index.train(product_features)
    index.add(product_features)
    
    # 내적 유사도 검색 (이미 정규화됨)
    D, I = index.search(query_feature, K)
    print("추천 인덱스 (npy):", I[0])
    
    # --- Combined similarity 계산 (feature, color, category) ---
    alpha = 0.6  # feature similarity weight
    beta = 0.3   # color similarity weight
    gamma = 0.1  # category similarity weight
    combined_scores = []
    for i in range(product_features.shape[0]):
        feat_sim = cosine_similarity(query_feature.flatten(), product_features[i])
        prod_idx, box_idx = map_index_to_json(i, products)
        if prod_idx is None:
            prod_cat_sim = 0
            prod_color_sim = 0
        else:
            prod_cloth = products[prod_idx]["clothes"][box_idx]
            prod_color = prod_cloth.get("color_vector")
            prod_cat = prod_cloth.get("category")
            # Color similarity
            if prod_color is None:
                prod_color_sim = 0
            else:
                prod_color = np.array(prod_color, dtype=float)
                norm_prod = np.linalg.norm(prod_color)
                if norm_prod == 0:
                    norm_prod = 1
                prod_color /= norm_prod
                prod_color_sim = cosine_similarity(query_color, prod_color)
            # Category similarity: 1 if match, else 0 (case-insensitive)
            if query_category is None or prod_cat is None:
                prod_cat_sim = 0
            else:
                prod_cat_sim = 1 if query_category.strip().lower() == prod_cat.strip().lower() else 0
        combined = alpha * feat_sim + beta * prod_color_sim + gamma * prod_cat_sim
        combined_scores.append(combined)
    combined_scores = np.array(combined_scores)
    
    # --- 상위 k개 제품 선택 ---
    sorted_indices = np.argsort(-combined_scores)
    unique_indices = []
    for idx in sorted_indices:
        if idx not in unique_indices:
            unique_indices.append(idx)
        if len(unique_indices) >= K:
            break
    print("추천 인덱스 (combined):", unique_indices)
    print("유사도 점수 (combined):", combined_scores[unique_indices])
    
    # --- npy 인덱스를 JSON의 제품 및 box 인덱스로 매핑 ---
    mapping = []
    for npy_idx in unique_indices:
        prod_idx, box_idx = map_index_to_json(npy_idx, products)
        if prod_idx is not None:
            mapping.append((npy_idx, prod_idx, box_idx))
    print("매핑 결과 (npy index, product index, box index):", mapping)
    
    # --- 추천 결과 이미지 출력 (제품명, score, category, HSV 색상 패치 표시) ---
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
            cv2.rectangle(prod_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 텍스트: 제품명, score, category
            text = f"{prod_name} | {score:.2f} | {prod_cat}"
            cv2.putText(prod_img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 색상 패치: JSON의 color_vector (HSV)을 BGR로 변환하여 출력
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
                cv2.putText(prod_img, color_text, (patch_x1, patch_y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        window_name = f"Product {prod_idx} Box {box_idx}"
        cv2.imshow(window_name, prod_img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()
