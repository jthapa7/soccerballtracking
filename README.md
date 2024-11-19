# Soccer Ball Tracking using Object Detection Models
This project implements a soccer ball tracking system using YOLO on a Raspberry Pi. The system uses a camera feed to detect and track a soccer ball, adjusting servo motors to keep the ball centered in the frame. This setup is ideal for sports recording applications, providing an automated solution to follow the game ball in real time.
Along the project I found out that running YOLO models on Raspberry Pi with out any external devices like Google Coral is painful since the detection speed is really slow. Therefore I first train the yolo model that detects the soccer ball then I exported the model to the TPU format using the follow code:
```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/content/runs/detect/yolov8n_custom3/weights/best.pt")

# Export the model to TFLite Edge TPU format
model.export(format="edgetpu")  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'
```
With this I was getting inference speed of 100ms from 2000ms.

# Create Yaml File using Python
```python
import yaml

# Define the configuration
yolo_config = {
    "train": "/content/dataset/train/images",
    "val": "/content/dataset/validate/images",
    "names": {
        0: "ball"
    }
}

# Specify the output file name
output_file = "data.yaml"

# Write the configuration to a YAML file with formatting
with open(output_file, "w") as file:
    yaml.dump(yolo_config, file, default_flow_style=False, sort_keys=False)
```
This will create yaml file `data.yaml` that we later use in training purpose.
# Training Yolo model

# 1. Get the dataset folder with training images and validate images with corresponding labels in .txt file using `LabelImg` app. The structure of the dataset folder should be as shown below
```
dataset/
|-----train/
|       |--images/
|       |--lables/ (.txt file that you get after labelling using LabelImg)   
|-----validate/
        |--images/
        |--labels/ (.txt file that you get after labelling using LabelImg)

```
# 2. Upload the Zip file to Collab Session. In collab you can use the GPU which is free that help to train the model faster than using your CPU.
# 3. In collab follow these steps:
## Step 1.
    Install `Ultralytics`
    ```
    !pip install ultralytics
    ```
## Step 2.
    Unzip the dataset
    ```
    !unzip /content/dataset.zip
    ```
## Step 3.
    Create a `data.yaml` file using the method above
## Step 4.
   Run the following code
```
from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt') #Yolo model that you want to train

# Training.
results = model.train(
   data='data.yaml',
   imgsz=640,     #Default Image size
   epochs=150,
   batch=8,
   name='yolov8n_custom' #Name of the folder to which you want to save your model to
)
```
## Step 5. (Optional)
You can export the custom model as Edge TPU format so that it runs faster. If you are trying to run YOLO models in Raspberry Pi then I recomment you use this.
```
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/content/runs/detect/yolov8n_custom3/weights/best.pt")

# Export the model to TFLite Edge TPU format
model.export(format="edgetpu")  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'
```
   
