"""
Eagle Eye VLM
"""

import torch
from torch import nn


class EagleEye(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor):
        """forward the tensor to model to predict"""
        return x

    def __call__(self, x: torch.Tensor):
        """internal call behavior"""
        print("internal call function is called")
        return self.forward(x)


eagle_eye = EagleEye()
