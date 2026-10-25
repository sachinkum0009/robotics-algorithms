import numpy as np
import torch
from torchvision import transforms


def generate_anchors(scales: list[float], ratios: list[float]):
    anchors: list[tuple[float, float]] = []
    for scale in scales:
        for ratio in ratios:
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchors.append((width, height))
    return anchors


def convert_to_yolo_format(
    width: int, height: int, bbox: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return (x_center, y_center, box_width, box_height)


train_transforms = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]
)


def yolo_loss(
    predictions,
    targets,
    num_classes,
    num_anchors=3,
    lambda_coord=5,
    lambda_noobj=0.5,
):
    B, S, _, total = (
        predictions.shape
    )  # total = num_anchors * (5 + num_classes)
    predictions = predictions.view(B, S, S, num_anchors, 5 + num_classes)

    # Expand targets from [B, S, S, 5+C] -> [B, S, S, num_anchors, 5+C]
    # by broadcasting across the anchor dimension
    targets = targets.unsqueeze(3).expand_as(predictions)

    pred_boxes = predictions[..., :4]
    pred_conf = predictions[..., 4]
    pred_classes = predictions[..., 5:]

    target_boxes = targets[..., :4]
    target_conf = targets[..., 4]
    target_classes = targets[..., 5:]

    obj_mask = target_conf == 1

    # Localization loss (only where object exists)
    box_loss = lambda_coord * torch.sum(
        (pred_boxes[obj_mask] - target_boxes[obj_mask]) ** 2
    )

    # Confidence loss
    obj_loss = torch.sum((pred_conf[obj_mask] - target_conf[obj_mask]) ** 2)
    noobj_loss = lambda_noobj * torch.sum(pred_conf[~obj_mask] ** 2)

    # Classification loss (only where object exists)
    class_loss = torch.sum(
        (pred_classes[obj_mask] - target_classes[obj_mask]) ** 2
    )

    total_loss = (
        box_loss + obj_loss + noobj_loss + class_loss
    ) / predictions.shape[0]

    return total_loss


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])  # was max — must be min for intersection corner
    y2 = min(box1[3], box2[3])  # same fix

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)  # epsilon prevents division by zero


# def yolo_loss(
#     predictions, targets, num_classes, lambda_coord=5, lambda_noobj=0.5
# ):
#     """
#     Compute YOLO loss
#     - predictions: Predicted tensor
#     - targets: Ground truth tensor
#     """
#     predictions = predictions.view(-1, 7, 7, 3, 25)
#     pred_boxes = predictions[..., :4]
#     pred_conf = predictions[..., 4]
#     pred_classes = predictions[..., 5:]
#     target_boxes = targets[..., :4]
#     target_conf = targets[..., 4]
#     target_classes = targets[..., 5:]

#     # Localization loss
#     box_loss = lambda_coord * torch.sum((pred_boxes - target_boxes) ** 2)

#     # Confidence Loss
#     obj_loss = torch.sum((pred_conf - target_conf) ** 2)
#     noobj_loss = lambda_noobj * torch.sum((pred_conf[target_conf == 0]) ** 2)

#     # Classification Loss
#     class_loss = torch.sum((pred_classes - target_classes) ** 2)

#     # Total Loss
#     total_loss = box_loss + obj_loss + noobj_loss + class_loss
#     return total_loss


# def compute_iou(box1, box2):
#     """Compute IoU between two bouding boxes."""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = max(box1[2], box2[2])
#     y2 = max(box1[3], box2[3])

#     intersection = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = box1_area + box2_area - intersection

#     return intersection / union


def calculation_map(predictions, targets, iou_threshold=0.5):
    """Calculates mean Average Precision"""
    average_precisions = []
    for c in range(num_classes):
        # Filter predictions and targets for class c
        pred_c = [p for p in predictions if p[0] == c]
        target_c = [t for t in targets if t[0] == c]

        # Sort predictions by confidence score
        pred_c.sort(key=lambda x: x[1], reverse=True)

        # Compute precision and recall
        precision = []
        recall = []
        for i, pred in enumerate(pred_c):
            iou = compute_iou(pred[2:], target_c[i][1:])
            if iou >= iou_threshold:
                precision.append(1)
                recall.append(1)
            else:
                precision.append(0)
                recall.append(0)

        # Calculate Average Precision
        average_precisions.append(sum(precision) / len(precision))
    return sum(average_precisions) / len(average_precisions)


def non_max_suppression(predictions, iou_threshold=0.5):
    """Applies NMS to suppress overlapping boxes."""
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    final_predictions = []

    while predictions:
        best = predictions.pop(0)
        predictions = [
            p
            for p in predictions
            if compute_iou(best[2:], p[2:]) < iou_threshold
        ]
        final_predictions.append(best)

    return final_predictions


DEVICE = (
    "mps"
    if torch.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
