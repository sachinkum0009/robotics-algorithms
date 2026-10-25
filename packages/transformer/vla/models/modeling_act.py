"""
Action Chunking Transformer

"""

import torch
from lerobot.policies.act.modeling_act import (
    ACT,
    ACTConfig,
    ACTDecoder,
    ACTEncoder,
    ACTPolicy,
)
from torch import nn
from torch.nn import functional as F


class ActModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

class ActionChunking(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x
