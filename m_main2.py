import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 여러 카메라의 번호를 리스트로 지정합니다.
camera_indices = [1, 2, 3, 4, 5, 6, 7, 8]  # 예시로 0, 1, 2번 카메라를 사용합니다. 필요한 개수만큼 추가하세요.

# 각각의 카메라에 대한 VideoCapture 객체를 생성합니다.
capture_objects = [cv2.VideoCapture(idx) for idx in camera_indices]

# 사전 훈련된 PyTorch 모델을 로드합니다.
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
model.eval()

# 전처리 파이프라인을 설정합니다.
preprocess = transforms.Compose([transforms.Resize((384, 384)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

try:
    while True:
        for i, cap in enumerate(capture_objects):
            # 각 카메라에서 프레임을 읽어옵니다.
            ret, frame = cap.read()

            # OpenCV BGR 이미지를 RGB로 변환합니다.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 이미지를 PIL 이미지로 변환하고 전처리합니다.
            input_image = Image.fromarray(rgb_frame)
            input_tensor = preprocess(input_image).unsqueeze(0)

            # 모델에 이미지를 전달하여 깊이 맵을 추정합니다.
            with torch.no_grad():
                prediction = model(input_tensor)

            # 추정된 깊이 맵을 가져옵니다.
            depth_map = prediction.squeeze().cpu().numpy()

            # 깊이에 따라 회색으로 표시합니다.
            min_depth = depth_map.min()
            max_depth = depth_map.max()
            normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
            gray_depth = (normalized_depth * 255).astype(np.uint8)

            # OpenCV를 사용하여 이미지를 윈도우에 표시합니다.
            window_name = f'Depth Map Camera {i}'
            cv2.imshow(window_name, gray_depth)

        # 'q' 키를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 모든 카메라 캡처를 해제하고 윈도우를 닫습니다.
    for cap in capture_objects:
        cap.release()
    cv2.destroyAllWindows()
