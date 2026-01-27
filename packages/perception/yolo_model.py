import torch
import torch.nn as nn
from ultralytics import YOLO #type: ignore


model = YOLO("yolo26n.pt")
model.export(format="onnx", dynamic=True)

