import cv2
import threading

# 카메라 디바이스 네임을 지정하여 카메라 열기
cap1 = cv2.VideoCapture(0)  # 카메라 1
cap2 = cv2.VideoCapture(3)  # 카메라 2
cap3 = cv2.VideoCapture(4)  # 카메라 3
cap4 = cv2.VideoCapture(5)  # 카메라 4
cap5 = cv2.VideoCapture(6)  # 카메라 5

# 카메라가 정상적으로 열렸는지 확인
if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened() and cap5.isOpened()):
    print("하나 이상의 카메라를 열 수 없습니다.")
    exit(1)

# 카메라를 처리할 스레드 함수 정의
def camera_thread(cap, camera_name):
    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        # 프레임이 정상적으로 읽혔는지 확인
        if not ret:
            print(f"{camera_name}: 프레임을 읽을 수 없습니다.")
            break

        # 카메라에서 읽은 프레임을 화면에 표시
        cv2.imshow(camera_name, frame)

        # 'q' 키를 누르면 스레드 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 사용이 끝난 카메라를 해제
    cap.release()

# 각 카메라에 대한 스레드 생성
thread1 = threading.Thread(target=camera_thread, args=(cap1, 'Camera 1'))
thread2 = threading.Thread(target=camera_thread, args=(cap2, 'Camera 2'))
thread3 = threading.Thread(target=camera_thread, args=(cap3, 'Camera 3'))
thread4 = threading.Thread(target=camera_thread, args=(cap4, 'Camera 4'))
thread5 = threading.Thread(target=camera_thread, args=(cap5, 'Camera 5'))

# 스레드 시작
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()

# 스레드가 종료될 때까지 대기
thread1.join()
thread2.join()
thread3.join()
thread4.join()
thread5.join()

# 모든 창 닫기
cv2.destroyAllWindows()
