import cv2

# 고유한 카메라 식별자 목록
camera_identifiers = [
    "USB\VID_4C4A&PID_4A55&REV_0100&MI_00",
    "USB\\VID_4C4A&PID_4A55&MI_00\\8&c182577&0&0000",
    "USB\\VID_4C4A&PID_4A55&MI_00\\8&195b3bd3&0&0000",
    "USB\\VID_4C4A&PID_4A55&MI_00\\8&2c14ec78&0&0000",
    "USB\\VID_4C4A&PID_4A55&MI_00\\7&1e948f5e&0&0000"
]

# 카메라를 저장할 변수 목록 초기화
cameras = []

# 각각의 고유한 식별자에 대해 카메라 열기
for identifier in camera_identifiers:
    cap = cv2.VideoCapture(f'DSHOW:{identifier}')
    if cap.isOpened():
        cameras.append(cap)
        print(f"카메라를 열었습니다: {identifier}")
    else:
        print(f"카메라를 열 수 없습니다: {identifier}")

# 카메라를 사용한 후 해제
for cap in cameras:
    cap.release()

# 창 모두 닫기
cv2.destroyAllWindows()
