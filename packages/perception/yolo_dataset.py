import os

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(
            self.label_dir, self.images[idx].replace(".jpg", ".txt")
        )

        # Load image
        image = cv2.imread(img_path)
        # if image:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )  # TODO: handle later

        # Load annotations
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_label, x, y, w, h])

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(boxes)
