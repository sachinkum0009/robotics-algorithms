"""
Inference YOLO
"""

import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from models.yolo import YOLO
from PIL import Image
from torchvision import transforms
from utils import DEVICE, VOC_CLASSES

model = YOLO(7, 20, 3)
state_dict = torch.load("yolo_model_100.pth", map_location=torch.device("mps"))
model.load_state_dict(state_dict)
# model.to(DEVICE)
model.eval()

img_path = "/Users/mac/zzzzz/python/robotics-algorithms/packages/perception/data/VOCdevkit/VOC2012/JPEGImages/2012_004312.jpg"
# img_path = "/Users/mac/zzzzz/python/robotics-algorithms/packages/perception/data/VOCdevkit/VOC2012/JPEGImages/2012_004147.jpg"
img = Image.open(img_path).convert("RGB")
origial_w, original_h = img.size

transform = transforms.Compose(
    [transforms.Resize((448, 448)), transforms.ToTensor()]
)
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    predictions = model(input_tensor)

predictions = predictions.view(7, 7, 3, 25)

detections = []
conf_threshold = 0.59

# 4. Iterate through the grid and anchors
for i in range(7):
    for j in range(7):
        for a in range(3):
            anchor_data = predictions[i, j, a]
            conf = torch.sigmoid(anchor_data[4]).item()  # Confidence score

            if conf > conf_threshold:
                # Get class with highest probability
                class_probs = torch.softmax(anchor_data[5:], dim=0)
                class_id = torch.argmax(class_probs).item()

                # Convert relative cell coords back to 0-1 image coords
                # x_cell, y_cell are relative to the top-left of the grid cell
                x_cell, y_cell, w_cell, h_cell = anchor_data[:4]

                x_img = (j + x_cell) / 7.0
                y_img = (i + y_cell) / 7.0
                w_img = w_cell / 7.0
                h_img = h_cell / 7.0

                detections.append(
                    {
                        "box": [x_img, y_img, w_img, h_img],
                        "conf": conf,
                        "class": VOC_CLASSES[class_id],
                    }
                )
fig, ax = plt.subplots(1)
ax.imshow(img)
w_orig, h_orig = img.size

for det in detections:
    x_c, y_c, w, h = det["box"]

    # Convert center-coords to top-left corner for Matplotlib
    xmin = (x_c - w / 2) * w_orig
    ymin = (y_c - h / 2) * h_orig
    width = w * w_orig
    height = h * h_orig

    rect = patches.Rectangle(
        (xmin, ymin),
        width,
        height,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.text(
        xmin,
        ymin,
        f"{det['class']} {det['conf']:.2f}",
        color="white",
        verticalalignment="top",
        bbox={"color": "red", "pad": 0},
    )

plt.show()
