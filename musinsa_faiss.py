import faiss
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.ops as ops

CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve dress",
    "vest", "sling", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress"
]

def load_model(model_path, device):
    """DF2MatchRCNN 모델 로드 및 Feature Extractor 설정"""
    # weights_only=True로 보안 경고 완화 (모델 파일이 신뢰할 수 있다고 가정)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=14)
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

def extract_roi_features(image_tensor, model, device, detected_boxes):
    """Faster R-CNN의 feature map에서 ROI Align을 이용하여 특징 벡터 추출"""
    with torch.no_grad():
        features = model.backbone(image_tensor.to(device))
    # 가장 해상도가 높은 feature map 선택 (보통 '0' 키)
    feature_map = features["0"]  # shape: (1, C, H_feat, W_feat)
    _, _, H_feat, _ = feature_map.shape
    _, _, H_img, _ = image_tensor.shape
    # 이미지와 feature map의 높이 비율을 spatial_scale로 사용
    spatial_scale = H_feat / H_img

    # ROI Align을 위해 detected_boxes를 텐서로 변환 (원본 이미지 좌표)
    boxes = torch.tensor(detected_boxes, dtype=torch.float32, device=device)
    aligned_features = ops.roi_align(
        feature_map,
        [boxes],             # RoIs는 리스트 형태로 전달
        output_size=(7, 7),
        spatial_scale=spatial_scale,
        sampling_ratio=2
    )  # 결과: (N, C, 7, 7)
    pooled_features = aligned_features.mean(dim=[2, 3])  # (N, C)
    return pooled_features.cpu().numpy()

def extract_clothes(image, model, device, score_threshold=0.5):
    """
    이미지에서 검출된 객체(의류)의 ROI 특징 벡터들을 추출한 후,
    각 ROI 벡터의 평균을 계산해 하나의 대표 벡터(query vector)로 반환합니다.
    """
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    
    # 점수가 높은 ROI만 선택
    selected_boxes = [box for box, score in zip(boxes, scores) if score >= score_threshold]
    if len(selected_boxes) == 0:
        # 검출된 ROI가 없으면 0 벡터 반환
        return np.zeros((1, 256), dtype=np.float32)
    
    roi_features = extract_roi_features(input_tensor, model, device, selected_boxes)
    # 여러 ROI 벡터들의 평균을 계산하여 하나의 query vector로 만듭니다.
    query_vector = roi_features.mean(axis=0, keepdims=True)  # shape: (1, 256)
    return query_vector

# === hoodie_with_feat.npy 파일 로드 및 전처리 ===
hoodie = np.load("hoodie_with_feat.npy", allow_pickle=True)
if not isinstance(hoodie, np.ndarray):
    hoodie = np.array(hoodie)
if not np.issubdtype(hoodie.dtype, np.floating):
    hoodie = hoodie.astype(np.float32)
if np.isnan(hoodie).any():
    hoodie = np.nan_to_num(hoodie)
if len(hoodie.shape) == 1:
    hoodie = hoodie.reshape(1, -1)
# 각 벡터를 정규화하여 단위 벡터로 만듭니다.
norms = np.linalg.norm(hoodie, axis=1, keepdims=True)
norms[norms == 0] = 1
hoodie /= norms

print("데이터 변환 완료. Shape:", hoodie.shape, "Type:", hoodie.dtype)

# === 모델 로드 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "df2matchrcnn"  # 모델 파일 경로
model = load_model(model_path, device)

# === IVF 인덱스 생성 (cosine similarity를 위해 inner product 사용) ===
d = hoodie.shape[1]  # 특징 벡터 차원 (예: 256)
nlist = 5            # 데이터가 228개이므로 작은 클러스터 개수 사용
# quantizer는 inner product 기반으로 생성
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
if not index.is_trained:
    index.train(hoodie)
index.add(hoodie)

# === 테스트 이미지 처리 및 검색 ===
test_img = Image.open("test.png").convert("RGB")
query_vector = extract_clothes(test_img, model, device)
if len(query_vector.shape) == 1:
    query_vector = query_vector.reshape(1, -1)
# Query vector도 단위 벡터로 정규화 (cosine similarity 계산을 위해)
qnorm = np.linalg.norm(query_vector, axis=1, keepdims=True)
qnorm[qnorm == 0] = 1
query_vector /= qnorm

# 디버그 출력: query vector와 hoodie의 첫 번째 벡터 간의 cosine similarity (내적)
cos_sim = np.dot(query_vector, hoodie[0].reshape(-1,1))
print("Query vector:", query_vector)
print("첫 번째 hoodie 벡터:", hoodie[0])
print("수동 cosine similarity:", cos_sim)

k = 5  # 가장 유사한 k개 검색
# 검색 결과: 내적 값이 cosine similarity로 반환됨 (1에 가까울수록 유사)
similarities, indices = index.search(query_vector, k)

print("추천 의류 인덱스:", indices)
print("유사도 점수 (cosine similarity):", similarities)
