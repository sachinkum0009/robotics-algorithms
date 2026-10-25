# from ultralytics import YOLO
import torch
from models.yolo import YOLO
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from utils import DEVICE, VOC_CLASSES, generate_anchors, yolo_loss
from yolo_dataset import YOLODataset

scales = [0.1, 0.2, 0.4]
ratios = [0.5, 1, 2]
anchors = generate_anchors(scales, ratios)

# 1. Define the transform
# YOLO usually requires a fixed input size (e.g., 448x448 or 640x640)
transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]
)

# 2. Download the Dataset
# This downloads VOC 2012 (the standard version)
voc_data = datasets.VOCDetection(
    root="data/",
    year="2012",
    image_set="train",
    download=True,
    transform=transform,
)


class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}


# 3. Important: The Label Converter
# torchvision returns a complex dictionary of XML data.
# You need a function to convert that to the [class, x, y, w, h] format
# that your YOLODataset or loss function uses.
# def voc_collate_fn(batch):
#     images = []
#     targets = []

#     for img, annot in batch:
#         images.append(img)
#         # Parse 'annot' (XML dict) here to match your specific loss function's needs
#         # Extract: annot['annotation']['object'] -> bndbox
#         targets.append(annot)

#     return torch.stack(images), targets


def voc_collate_fn(batch):
    images = []
    # S=7 is based on your model = YOLO(7, 20, 3) where 7 is the grid size
    S = 7
    C = 20
    batch_targets = []

    for img, annot in batch:
        images.append(img)

        # Create an empty target grid: [S, S, 5 + C]
        # 5 elements are [x, y, w, h, confidence]
        target_grid = torch.zeros((S, S, 5 + C))

        objs = annot["annotation"]["object"]
        # Handle cases with single vs multiple objects
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            class_idx = class_to_idx[obj["name"]]
            bbox = obj["bndbox"]

            # 1. Get absolute pixel coordinates
            # VOC uses 1-based indexing for pixels
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            # 2. Get image dimensions (from the XML)
            img_w = float(annot["annotation"]["size"]["width"])
            img_h = float(annot["annotation"]["size"]["height"])

            # 3. Convert to YOLO format (0-1 relative to image)
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            # 4. Determine which grid cell this object belongs to
            i, j = int(S * y_center), int(S * x_center)

            # 5. If cell is empty, assign object (YOLOv1 logic)
            if target_grid[i, j, 4] == 0:
                target_grid[i, j, 4] = 1  # Confidence
                # Coordinates relative to cell
                x_cell = S * x_center - j
                y_cell = S * y_center - i
                w_cell = S * width
                h_cell = S * height

                target_grid[i, j, :4] = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )
                target_grid[i, j, 5 + class_idx] = 1  # One-hot class

        batch_targets.append(target_grid)

    return torch.stack(images), torch.stack(batch_targets)


train_loader = DataLoader(
    voc_data, batch_size=8, shuffle=True, collate_fn=voc_collate_fn
)


# model = YOLO("yolo26n.pt")
# model.export(format="onnx", dynamic=True)
model = YOLO(7, 20, 3)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = yolo_loss
# print(model)

# train_dataset = YOLODataset(
#     img_dir="data/images", label_dir="data/labels", transforms=ToTensor()
# )
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        predictions = model(images)

        # Loss calculation
        loss = criterion(predictions, targets, num_classes=20)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "yolo_model_weights2.pth")
