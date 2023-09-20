import cv2
import numpy as np

# Load YOLOv4 model and weights
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Load YOLO class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load image
image = cv2.imread('image.png')

# Get image dimensions
height, width = image.shape[:2]

# Create a blob from the image and set input
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get the output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Run forward pass
detection_results = net.forward(output_layer_names)

# Initialize lists to store detected class IDs, confidences, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Minimum confidence threshold for detections
conf_threshold = 0.5

# Non-maximum suppression threshold
nms_threshold = 0.4

# Loop through each detection result
for result in detection_results:
    for detection in result:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            # Scale the bounding box coordinates back to the original image
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# # Draw the bounding boxes and labels on the image
# for i in indices:
#     i = i[0]
#     box = boxes[i]
#     x, y, w, h = box
#     label = str(classes[class_ids[i]])
#     confidence = confidences[i]
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Draw the bounding boxes and labels on the image
for i in indices:
    # i = i[0]  # This line should be removed
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
