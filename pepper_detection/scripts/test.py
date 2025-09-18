import os
import torch
import cv2

from ultralytics import YOLO
from ultralytics.engine.results import Results


IMAGE_NAME = "test11.jpg"


# Check if CUDA is available
try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except AssertionError:
    device = torch.device("cpu")


model_path = os.path.join(os.getcwd(), "models/prototype/best.pt")
image_path = os.path.join(os.getcwd(), f"data/test_image/{IMAGE_NAME}")
output_path = os.path.join(os.getcwd(), "output")

model = YOLO(model_path)
results: Results = model(image_path, device=device, conf=0.6)

for r in results:
    cv2.imshow("Output", r.plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
