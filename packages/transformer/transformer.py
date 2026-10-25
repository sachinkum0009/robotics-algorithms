"""
Implementation of transformer
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiHeadAttenion(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """Multi Head Attention to focus on different parts of
        the sentence at the same time"""
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, mask=None
    ):
        attn_score = torch.matmul(
            Q,
            K.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        attn_probe = F.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_probe, V)
        return output

    def split_head(self, x: Tensor):
        batch_size, seq_length, d_model = x.size()
        return x.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, head_dim = x.size()
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.d_model)
        )

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None):
        Q = self.split_head(self.W_q(Q))
        K = self.split_head(self.W_k(K))
        V = self.split_head(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x
