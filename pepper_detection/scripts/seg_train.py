import os
import torch
from ultralytics import YOLO

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())


device = torch.device("cuda:0")

model_file_path = os.path.join(
    os.getcwd(), "models/pretrained/yolo11l-seg.pt")
print(model_file_path)

# # Load a model
model = YOLO(model_file_path)

# # Load a model from a custom path
data_file_path = os.path.join(os.getcwd(), "data/pepper/data.yaml")

# # Train the model
train_results = model.train(
    data=data_file_path,  # data.yaml file
    epochs=300,  # number of training epochs
    imgsz=640,  # training image size
    device=device,  # MPS device
)

# Evaluate model performance on the validation set
metrics = model.val()
