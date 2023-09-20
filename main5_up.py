import cv2

# 카메라 디바이스 네임을 지정하여 카메라 열기
cap1 = cv2.VideoCapture(0)  # 카메라 1
cap2 = cv2.VideoCapture(3)  # 카메라 2
cap3 = cv2.VideoCapture(4)  # 카메라 3
cap4 = cv2.VideoCapture(5)  # 카메라 4
cap5 = cv2.VideoCapture(6)  # 카메라 5

#    4   5
# 3         6
#
#
#      0

# 카메라가 정상적으로 열렸는지 확인
if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened() and cap5.isOpened()):
    print("하나 이상의 카메라를 열 수 없습니다.")

while True:
    # 프레임 읽기
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    ret5, frame5 = cap5.read()

    # 프레임이 정상적으로 읽혔는지 확인
    if not (ret1 and ret2 and ret3 and ret4 and ret5):
        print("프레임을 읽을 수 없습니다.")
        break

    # 각각의 카메라에서 읽은 프레임을 화면에 표시
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)
    cv2.imshow('Camera 3', frame3)
    cv2.imshow('Camera 4', frame4)
    cv2.imshow('Camera 5', frame5)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용이 끝난 카메라를 해제
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cap5.release()

# 창 모두 닫기
cv2.destroyAllWindows()
