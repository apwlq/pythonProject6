import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 2번과 3번 카메라를 초기화합니다.
cap1 = cv2.VideoCapture(2)  # 2번 카메라
cap2 = cv2.VideoCapture(3)  # 3번 카메라

# 사전 훈련된 PyTorch 모델을 로드합니다.
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
model.eval()

# 전처리 파이프라인을 설정합니다.
preprocess = transforms.Compose([transforms.Resize((384, 384)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

while True:
    # 두 개의 카메라에서 프레임을 읽어옵니다.
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # 각각의 프레임을 OpenCV BGR 이미지에서 RGB로 변환합니다.
    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # 이미지를 PIL 이미지로 변환하고 전처리합니다.
    input_image1 = Image.fromarray(rgb_frame1)
    input_image2 = Image.fromarray(rgb_frame2)
    input_tensor1 = preprocess(input_image1).unsqueeze(0)
    input_tensor2 = preprocess(input_image2).unsqueeze(0)

    # 모델에 이미지를 전달하여 깊이 맵을 추정합니다.
    with torch.no_grad():
        prediction1 = model(input_tensor1)
        prediction2 = model(input_tensor2)

    # 추정된 깊이 맵을 가져옵니다.
    depth_map1 = prediction1.squeeze().cpu().numpy()
    depth_map2 = prediction2.squeeze().cpu().numpy()

    # 깊이에 따라 회색으로 표시합니다.
    min_depth1 = depth_map1.min()
    max_depth1 = depth_map1.max()
    normalized_depth1 = (depth_map1 - min_depth1) / (max_depth1 - min_depth1)
    gray_depth1 = (normalized_depth1 * 255).astype(np.uint8)

    min_depth2 = depth_map2.min()
    max_depth2 = depth_map2.max()
    normalized_depth2 = (depth_map2 - min_depth2) / (max_depth2 - min_depth2)
    gray_depth2 = (normalized_depth2 * 255).astype(np.uint8)

    # OpenCV를 사용하여 이미지를 윈도우에 표시합니다.
    cv2.imshow('Depth Map Camera 2', gray_depth1)
    cv2.imshow('Depth Map Camera 3', gray_depth2)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 캡처를 해제하고 윈도우를 닫습니다.
cap1.release()
cap2.release()
cv2.destroyAllWindows()
