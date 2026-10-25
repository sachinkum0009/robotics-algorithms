"""
YOLO (You Only Look Once)
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """A block of Conv2D -> BatchNorm -> ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class YOLOBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 448 -> 224
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 28 -> 14
            ConvBlock(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 14 -> 7  ✓ matches grid_size=7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class YOLOHead(nn.Module):
    def __init__(self, grid_size: int, num_classes: int, num_anchors: int):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(
            512, num_anchors * (5 + num_classes), kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()


class YOLO(nn.Module):
    def __init__(
        self, grid_zie: int = 7, num_classes: int = 20, num_anchors: int = 3
    ):
        super().__init__()
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(grid_zie, num_classes, num_anchors)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions


class YOLO_Old(nn.Module):
    def __init__(self, num_classes=20, num_anchers=3, grid_size=7):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchers = num_anchers
        self.grid_size = grid_size

        # Backbone: Feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Detection Head: Outputs bounding boxes, confidence scores, and class probabilities
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                256 * (grid_size // 4) ** 2,
                grid_size * grid_size * (num_anchers * 5 + num_classes),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        predictions = self.detector(features)
        return predictions.view(
            -1,
            self.grid_size,
            self.grid_size,
            self.num_anchers * 5 + self.num_classes,
        )
