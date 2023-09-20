import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import asyncio

# Load YOLOv4 model and weights
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
frame_width = 384
frame_height = 384

# Load YOLO class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# 카메라 캡처를 초기화합니다.
cap = cv2.VideoCapture(0)  # 0은 내장 웹캠을 나타냅니다. 다른 카메라 장치를 사용하려면 인덱스를 조정하세요.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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
    colored_depth = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    colored_depth[:, :, 0] = (1.0 - normalized_depth) * 255  # 빨강 채널
    colored_depth[:, :, 1] = (0.5 - normalized_depth) * 125  # 녹색 채널
    colored_depth[:, :, 2] = normalized_depth * 255  # 파랑 채널

    # Create a blob from the frame and set input for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names for YOLO
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass for YOLO
    detection_results = net.forward(output_layer_names)

    # Initialize lists to store detected class IDs, confidences, and bounding boxes for YOLO
    class_ids = []
    confidences = []
    boxes = []

    # Minimum confidence threshold for YOLO detections
    conf_threshold = 0.5

    # Non-maximum suppression threshold for YOLO
    nms_threshold = 0.4

    # Loop through each YOLO detection result
    for result in detection_results:
        for detection in result:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Scale the bounding box coordinates back to the original frame
                center_x = int(detection[0] * gray_depth.shape[1])
                center_y = int(detection[1] * gray_depth.shape[0])
                w = int(detection[2] * gray_depth.shape[1])
                h = int(detection[3] * gray_depth.shape[0])

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping boxes for YOLO
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw the bounding boxes and labels on the frame for YOLO
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(gray_depth, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(gray_depth, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    # cv2.imshow('Object Detection and Depth Map', frame)
    cv2.imshow('Depth Map', gray_depth)
    # cv2.imshow('Colored Depth Map', colored_depth)
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
