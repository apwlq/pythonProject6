import cv2
import multiprocessing

# 함수를 사용하여 카메라 스트림을 읽는 프로세스 정의
def capture_camera(camera_index, frame_queue):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라 {camera_index}을(를) 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"카메라 {camera_index}의 프레임을 읽을 수 없습니다.")
            break
        frame_queue.put((camera_index, frame))

    cap.release()

if __name__ == "__main__":
    num_cameras = 8
    frame_queue = multiprocessing.Queue()

    # 각 카메라에 대한 프로세스 생성 및 시작
    processes = []
    for i in range(num_cameras):
        p = multiprocessing.Process(target=capture_camera, args=(i, frame_queue))
        p.start()
        processes.append(p)

    while True:
        frames = {}
        for _ in range(num_cameras):
            camera_index, frame = frame_queue.get()
            frames[camera_index] = frame

        if len(frames) == num_cameras:
            combined_frames = [frames[i] for i in range(num_cameras)]
            combined_frame = cv2.hconcat(combined_frames)
            cv2.imshow('Multiple Cameras', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 모든 프로세스를 종료
    for p in processes:
        p.terminate()

    cv2.destroyAllWindows()
