import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-base")
model = AutoModel.from_pretrained("facebook/dinov3-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)  # type: ignore

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(f"DINOv3 feature shape: {last_hidden_states.shape}")
