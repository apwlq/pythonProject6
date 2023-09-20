import cv2

# 카메라 디바이스 네임을 지정하여 카메라 열기
cap1 = cv2.VideoCapture(1)  # 카메라 1
cap2 = cv2.VideoCapture(2)  # 카메라 2
cap3 = cv2.VideoCapture(3)  # 카메라 3
cap4 = cv2.VideoCapture(4)  # 카메라 4
cap5 = cv2.VideoCapture(5)  # 카메라 5

# 카메라가 정상적으로 열렸는지 확인
if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened() and cap5.isOpened()):
    print("하나 이상의 카메라를 열 수 없습니다.")

# 카메라 사용 후에는 해제
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cap5.release()
