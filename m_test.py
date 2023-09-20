import cv2

for i in range(10):  # 예: 0부터 9까지의 카메라 인덱스를 확인
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available.")
        cap.release()
