import torch.nn as nn
import numpy as np

multihead_attention = nn.MultiheadAttention(embed_dim=512, num_heads=1)

in_proj_weight = multihead_attention.in_proj_weight

print("in_proj_weight shape:", in_proj_weight.shape)