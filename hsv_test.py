import cv2
import numpy as np

def hsv_to_bgr(h, s, v):
    """ HSV 값을 BGR로 변환 """
    hsv_color = np.uint8([[[h, s, v]]])  # HSV 값 생성
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)  # BGR로 변환
    return tuple(int(i) for i in bgr_color[0][0])  # BGR 값 반환

def show_color(h, s, v):
    """ HSV 값을 기반으로 OpenCV 창에서 색상 출력 """
    bgr_color = hsv_to_bgr(h, s, v)  # BGR 값 변환
    color_img = np.full((200, 400, 3), bgr_color, dtype=np.uint8)  # 색상 채운 이미지 생성
    
    # 창에 색상 표시
    cv2.imshow(f"Color - HSV({h}, {s}, {v})", color_img)
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()  # 창 닫기

if __name__ == "__main__":
    # 사용자 입력 받기
    h = int(input("Hue (0-179): "))
    s = int(input("Saturation (0-255): "))
    v = int(input("Value (0-255): "))

    # 입력값 검증
    if 0 <= h <= 179 and 0 <= s <= 255 and 0 <= v <= 255:
        show_color(h, s, v)
    else:
        print("HSV 값 범위를 확인하세요! (H: 0-179, S: 0-255, V: 0-255)")
