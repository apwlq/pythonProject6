import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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

while True:
    # 카메라에서 프레임을 읽어옵니다.
    ret, frame = cap.read()

    # OpenCV BGR 이미지를 RGB로 변환합니다.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 이미지를 PIL 이미지로 변환하고 전처리합니다.
    input_image = Image.fromarray(rgb_frame)
    input_tensor = preprocess(input_image).unsqueeze(0)

    # 모델에 이미지를 전달하여 깊이 맵을 추정합니다.
    with torch.no_grad():
        prediction = model(input_tensor)

    # 추정된 깊이 맵을 시각화합니다.
    depth_map = prediction.squeeze().cpu().numpy()
    plt.imshow(depth_map, cmap='inferno')
    plt.colorbar()
    plt.draw()
    plt.pause(0.01)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 캡처를 해제합니다.
cap.release()
cv2.destroyAllWindows()