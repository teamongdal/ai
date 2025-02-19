import faiss
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.ops as ops
import cv2
import logging

# df2matchrcnn 모델의 카테고리 리스트 (DeepFashion2 기준)
CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress",
    "trousers", "skirt", "shorts"
]
# HRNet에서 예측할 keypoint 개수 (모델 설정에 맞게 변경)
NUM_KEYPOINTS = 17

###############################################
# 모델 로드 함수들
###############################################
def load_detection_model(model_path, device):
    """df2matchrcnn 모델 로드 (객체 검출용)"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    from torchvision.models.detection import maskrcnn_resnet50_fpn
    new_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    # DeepFashion2의 경우 num_classes가 14 (배경 제외 13개 클래스 등)로 설정되어 있음
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=14)
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

def load_hrnet_model(model_path, device):
    """HRNet 모델 로드 (landmark 예측용)
       hrnet_deepfashion2.pth 파일은 state_dict만 포함하므로,
       hrnet.py에 정의된 HRNet 아키텍처를 import하여 모델 인스턴스를 생성한 후 state_dict를 로드합니다.
    """
    from hrnet import get_pose_net
    from easydict import EasyDict as edict

    cfg = edict({
        'MODEL': {
            'NAME': 'pose_hrnet',
            'NUM_JOINTS': NUM_KEYPOINTS,
            'TARGET_TYPE': 'gaussian',
            'INIT_WEIGHTS': False,
            'PRETRAINED': '',
            'EXTRA': {
                'FINAL_CONV_KERNEL': 1,
                'STAGE2': {
                    'NUM_MODULES': 1,
                    'NUM_BRANCHES': 2,
                    'NUM_BLOCKS': [4, 4],
                    'NUM_CHANNELS': [48, 96],
                    'BLOCK': 'BASIC',
                    'FUSE_METHOD': 'SUM'
                },
                'STAGE3': {
                    'NUM_MODULES': 4,
                    'NUM_BRANCHES': 3,
                    'NUM_BLOCKS': [4, 4, 4],
                    'NUM_CHANNELS': [48, 96, 192],
                    'BLOCK': 'BASIC',
                    'FUSE_METHOD': 'SUM'
                },
                'STAGE4': {
                    'NUM_MODULES': 3,
                    'NUM_BRANCHES': 4,
                    'NUM_BLOCKS': [4, 4, 4, 4],
                    'NUM_CHANNELS': [48, 96, 192, 384],
                    'BLOCK': 'BASIC',
                    'FUSE_METHOD': 'SUM'
                },
                'PRETRAINED_LAYERS': ['*']
            }
        }
    })

    model = get_pose_net(cfg, is_train=False)
    state_dict = torch.load(model_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # final_layer 파라미터 사이즈 불일치 시 제거하여 로드하도록 수정
    for key in list(state_dict.keys()):
        if key.startswith("final_layer") and key in model.state_dict():
            if state_dict[key].size() != model.state_dict()[key].size():
                print(f"Removing {key} from state_dict due to size mismatch: {state_dict[key].size()} vs {model.state_dict()[key].size()}")
                del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

###############################################
# 검출 및 랜드마크 함수들
###############################################
def detect_objects(image, model, device, score_threshold=0.5):
    """
    df2matchrcnn 모델을 이용하여 이미지에서 의류 객체(경계 상자, score, label)를 검출합니다.
    """
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_labels.append(label)
    return np.array(filtered_boxes), np.array(filtered_scores), np.array(filtered_labels)

def detect_landmarks(cropped_img, hrnet_model, device):
    """
    HRNet 모델을 이용하여 cropped_img (PIL 이미지)에서 keypoint heatmap을 예측하고,
    각 heatmap에서 최대 응답 위치를 찾아 cropped 영역 내의 랜드마크 좌표를 반환합니다.
    입력 이미지는 HRNet의 내부 연산에 맞도록, 원본 크기를 32의 배수로 리사이즈한 후 처리합니다.
    """
    import math
    # 원본 크기
    orig_w, orig_h = cropped_img.size
    # target size: 원본 크기를 넘는 가장 작은 32의 배수로 설정
    target_w = math.ceil(orig_w / 32) * 32
    target_h = math.ceil(orig_h / 32) * 32
    # cropped 이미지를 target size로 리사이즈 (BILINEAR interpolation)
    resized_img = cropped_img.resize((target_w, target_h), resample=Image.BILINEAR)
    
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(resized_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = hrnet_model(input_tensor)
    # output: (1, num_keypoints, H_out, W_out)
    heatmaps = output.squeeze(0)  # (num_keypoints, H_out, W_out)
    num_keypoints, H_out, W_out = heatmaps.shape
    factor_x = target_w / W_out
    factor_y = target_h / H_out
    
    landmarks = []
    for i in range(num_keypoints):
        heatmap = heatmaps[i]
        pos = torch.argmax(heatmap.view(-1))
        pos_y = pos // W_out
        pos_x = pos % W_out
        x = pos_x.float() * factor_x
        y = pos_y.float() * factor_y
        # 리사이즈한 좌표를 원본 크기로 다시 매핑
        x = x * (orig_w / target_w)
        y = y * (orig_h / target_h)
        landmarks.append((x.item(), y.item()))
    return np.array(landmarks)

def visualize_detections(pil_image, boxes, scores, labels, landmarks_list, score_threshold=0.5):
    """
    PIL 이미지에 검출된 bounding box, score, 카테고리 이름과 각 객체의 랜드마크를 표시하고 OpenCV 창으로 출력합니다.
    landmarks_list는 각 검출 영역에 대한 랜드마크 배열의 리스트여야 합니다.
    """
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        category = CATEGORIES[int(label) - 1] if (int(label) - 1) < len(CATEGORIES) else "unknown"
        text = f"{category}: {score:.2f}"
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        # 랜드마크 시각화: 검출된 영역 내 HRNet 예측 랜드마크 (전체 이미지 좌표로 변환)
        if idx < len(landmarks_list) and landmarks_list[idx] is not None:
            for (lx, ly) in landmarks_list[idx]:
                cv2.circle(image_cv, (int(x1 + lx), int(y1 + ly)), 3, (0, 0, 255), -1)
    
    cv2.imshow("Detections & Landmarks", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###############################################
# 모델 로드 및 검출, 랜드마크 추정 실행
###############################################
device = "cuda" if torch.cuda.is_available() else "cpu"

# df2matchrcnn 기반 객체 검출 모델 로드
det_model_path = "df2matchrcnn"  # df2matchrcnn 모델 파일 경로
det_model = load_detection_model(det_model_path, device)

# HRNet 기반 랜드마크 추정 모델 로드 (hrnet.py에 정의된 HRNet 아키텍처 사용)
hrnet_model_path = "hrnet_deepfashion2.pth"  # HRNet 모델 state_dict 파일 경로
hrnet_model = load_hrnet_model(hrnet_model_path, device)

# 테스트 이미지 로드
test_img = Image.open("zara_w-onepiece_0193_01.jpg").convert("RGB")
boxes, scores, labels = detect_objects(test_img, det_model, device, score_threshold=0.5)

# 각 검출 영역에 대해 HRNet으로 랜드마크 추정 (bounding box 내 좌표)
landmarks_list = []
for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    cropped = test_img.crop((x1, y1, x2, y2))
    lm = detect_landmarks(cropped, hrnet_model, device)
    landmarks_list.append(lm)

# 결과 시각화: 검출된 객체와 랜드마크 표시
visualize_detections(test_img, boxes, scores, labels, landmarks_list, score_threshold=0.5)
