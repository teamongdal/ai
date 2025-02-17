import json
import requests
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

def load_maskrcnn(model_path, device):
    # ✅ Mask R-CNN 모델 초기화 (14개의 클래스로 설정)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=14)

    # ✅ 모델 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)

    # ✅ Multi-GPU 학습 모델일 경우 "module." prefix 제거
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # ✅ 모델 로드 (strict=False로 일부 키가 누락되어도 로드)
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


# ✅ JSON 파일 로드
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ✅ JSON에서 `detail_url`로 해당 상품 찾기
def find_item_by_detail_url(json_data, detail_url):
    for item in json_data:
        if "detail_url" in item and item["detail_url"] == detail_url:
            return item
    return None

# ✅ 이미지 다운로드
def download_image(image_url):
    response = requests.get(image_url, stream=True, timeout=5)
    if response.status_code == 200:
        return Image.open(response.raw).convert("RGB")
    else:
        print(f"⚠ 이미지 다운로드 실패: {image_url}")
        return None

# ✅ Mask R-CNN을 사용하여 객체 탐지 및 마스크 생성
def detect_objects(image, model, device):
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(input_tensor)

    return predictions[0]

# ✅ 바운딩 박스 및 마스크 시각화
def visualize_results(image, predictions, clothes_data):
    image_np = np.array(image, dtype=np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # OpenCV용 BGR 변환
    overlay = image_np.copy()

    # JSON 데이터에서 바운딩 박스 정보 가져와 그리기
    for cloth in clothes_data:
        x1, y1, x2, y2 = map(int, cloth["box"])
        category = cloth["category"]

        # 바운딩 박스 그리기
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image_np, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ✅ Mask R-CNN의 마스크를 이미지 위에 적용
    for i, (box, score, mask) in enumerate(zip(predictions['boxes'], predictions['scores'], predictions['masks'])):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.tolist())
            mask = mask[0, :, :].detach().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255  # 마스크 이진화

            # ✅ 마스크 반투명하게 적용
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array(color) * 0.5  # 반투명 효과

            # ✅ 마스크 위에 바운딩 박스 그리기
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{category}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ 결과 이미지 출력
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# ✅ HSV Color Vector 시각화
def show_hsv_color(color_vector):
    hsv_color = np.uint8([[color_vector]])  # HSV 색상 벡터
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

    color_patch = np.full((100, 300, 3), bgr_color, dtype=np.uint8)  # 색상 패치 생성
    cv2.imshow("HSV Color Representation", color_patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ✅ 실행 함수
def main():
    json_path = "hoodie_with_cat&color.json"
    model_path = "df2matchrcnn"  # Mask R-CNN 모델 경로
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Mask R-CNN 모델 로드
    model = load_maskrcnn(model_path, device)

    # ✅ JSON 로드
    json_data = load_json(json_path)

    # ✅ 사용자 입력 (detail_url)
    detail_url = input("🔍 Enter the `detail_url`: ").strip()
    item = find_item_by_detail_url(json_data, detail_url)

    if not item:
        print(f"❌ 해당 `detail_url`({detail_url})을 찾을 수 없습니다.")
        return

    image_url = item["product_images_1"]
    clothes_data = item["clothes"]

    print(f"✅ Found Image URL: {image_url}")

    # ✅ 이미지 다운로드
    image = download_image(image_url)
    if image is None:
        return

    # ✅ Mask R-CNN으로 객체 탐지 및 마스크 생성
    predictions = detect_objects(image, model, device)

    # ✅ 바운딩 박스 및 마스크 시각화
    visualize_results(image, predictions, clothes_data)

    # ✅ HSV Color Vector 시각화
    for cloth in clothes_data:
        color_vector = cloth["color_vector"]
        print(f"🔹 Category: {cloth['category']} | Color Vector (HSV): {color_vector}")
        show_hsv_color(color_vector)

# ✅ 실행
if __name__ == "__main__":
    main()
