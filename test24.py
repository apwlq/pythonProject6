import cv2

# 카메라 1 열기
cap1 = cv2.VideoCapture(3, cv2.CAP_DSHOW)  # 카메라 인덱스 0번을 사용하며, DirectShow를 사용

# 카메라 2 열기
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # 카메라 인덱스 1번을 사용하며, DirectShow를 사용

# 카메라가 정상적으로 열렸는지 확인
if not cap1.isOpened() or not cap2.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # 프레임이 정상적으로 읽혔는지 확인
    if not ret1 or not ret2:
        print("프레임을 읽을 수 없습니다.")
        break

    # 각각의 카메라에서 읽은 프레임을 화면에 표시
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용이 끝난 카메라를 해제
cap1.release()
cap2.release()

# 창 닫기
cv2.destroyAllWindows()
