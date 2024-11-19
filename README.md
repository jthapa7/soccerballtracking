# Soccer Ball Tracking using Object Detection Models
This project implements a soccer ball tracking system using YOLOv5 on a Raspberry Pi. The system uses a camera feed to detect and track a soccer ball, adjusting servo motors to keep the ball centered in the frame. This setup is ideal for sports recording applications, providing an automated solution to follow the game ball in real time.
Along the project I found out that running YOLO models on Raspberry Pi with out any external devices like Google Coral is painful since the detection speed is really slow. Therefore I first train the yolo model that detects the soccer ball then I exported the model to the TPU format using the follow code:
```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/content/runs/detect/yolov8n_custom3/weights/best.pt")

# Export the model to TFLite Edge TPU format
model.export(format="edgetpu")  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'
```
With this I was getting inference speed of 100ms from 2000ms.
