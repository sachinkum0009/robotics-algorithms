import torch
import torch.nn as nn

# Define a Transformer model
transformer = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
)

# Example input
src = torch.rand((10, 32, 512))  # (seq_len, batch_size, d_model)
tgt = torch.rand((20, 32, 512))  # (seq_len, batch_size, d_model)

# Forward pass
output = transformer(src, tgt)
print(output.shape)
