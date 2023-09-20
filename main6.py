import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import asyncio

# 카메라 캡처를 초기화합니다.
cap = cv2.VideoCapture(0)  # 0은 내장 웹캠을 나타냅니다. 다른 카메라 장치를 사용하려면 인덱스를 조정하세요.

# 사전 훈련된 PyTorch 모델을 로드합니다.
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
model.eval()

# 전처리 파이프라인을 설정합니다.
preprocess = transforms.Compose([transforms.Resize((384, 384)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

async def process_frame(frame):
    # OpenCV BGR 이미지를 RGB로 변환합니다.
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    # 깊이에 따라 색상을 지정합니다.
    min_depth = depth_map.min()
    max_depth = depth_map.max()
    # normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
    colored_depth = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    colored_depth[:, :, 0] = (1.0 - normalized_depth) * 255  # 빨강 채널
    colored_depth[:, :, 1] = (0.5 - normalized_depth) * 125  # 녹색 채널
    colored_depth[:, :, 2] = normalized_depth * 255  # 파랑 채널

    # OpenCV를 사용하여 이미지를 윈도우에 표시합니다.
    cv2.imshow('Depth Map', gray_depth)
    cv2.imshow('Depth Map2', colored_depth)
    # cv2.imshow('origin1', rgb_frame)
    cv2.waitKey(1)

async def main():
    while True:
        # 카메라에서 프레임을 읽어옵니다.
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 처리를 비동기로 실행합니다.
        await asyncio.gather(process_frame(frame))

asyncio.run(main())

# 카메라 캡처를 해제하고 윈도우를 닫습니다.
cap.release()
cv2.destroyAllWindows()
