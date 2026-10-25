"""
UNet
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: Tensor):
        return self.decoder(self.encoder(x))
