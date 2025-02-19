import numpy as np
from PIL import Image
import cv2  # OpenCV 임포트
from roboflow import Roboflow

# Roboflow API 키 (자신의 API 키로 변경 필요)
ROBOFLOW_API_KEY = "iaKZpe4SwjpsNWkYh7aO"

CATEGORIES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress",
    "trousers", "skirt", "shorts",
]

###############################################
# Roboflow 모델 로드
###############################################
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("deepfashion2-m-11k")
model = project.version(1).model  # 모델 버전 변경 가능

###############################################
# Roboflow 모델을 사용한 객체 검출
###############################################
def detect_objects(image, model, score_threshold=0.5):
    """
    Roboflow 모델을 사용하여 이미지에서 패션 아이템 검출.
    반환: numpy 배열 (bounding boxes, scores, labels)
    """
    # 이미지를 저장하여 API 요청 (Roboflow 모델이 URL 또는 파일 경로 기반으로 예측 수행)
    image_path = "temp_image.jpg"
    image.save(image_path)  # PIL 이미지 저장

    # 모델을 사용하여 예측
    predictions = model.predict(image_path, confidence=score_threshold).json()

    boxes = []
    scores = []
    labels = []

    # Roboflow의 JSON 예측 결과에서 bbox 추출
    for pred in predictions["predictions"]:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        score = pred["confidence"]
        category = pred["class"]

        # 좌표 변환 (x_center, y_center, width, height) -> (x1, y1, x2, y2)
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        labels.append(category)

    return np.array(boxes), np.array(scores), labels

###############################################
# 검출 결과 시각화
###############################################
def visualize_detections(pil_image, boxes, scores, labels, score_threshold=0.5):
    """
    PIL 이미지에 검출된 bounding box, score, 카테고리 이름을 표시하고 OpenCV 창으로 출력.
    """
    # PIL 이미지를 OpenCV 이미지로 변환 (BGR 형식)
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        
        # 카테고리 찾기
        category = label if label in CATEGORIES else "unknown"
        text = f"{category}: {score:.2f}"

        # 초록색 Bounding Box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # 검출 결과 출력
    cv2.imshow("Detections", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###############################################
# 실행 코드 (테스트 이미지에 적용)
###############################################
test_img = Image.open("zara_w-onepiece_0193_01.jpg").convert("RGB")

# Roboflow 모델로 객체 검출
boxes, scores, labels = detect_objects(test_img, model, score_threshold=0.5)

# 검출된 객체 시각화
visualize_detections(test_img, boxes, scores, labels, score_threshold=0.5)
