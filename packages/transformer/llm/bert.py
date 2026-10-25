"""
BERT
"""

import torch
from torch import nn
from transformers import (
    RTDetrConfig,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
)


class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential()
