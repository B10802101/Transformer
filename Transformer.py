import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt

Seq = 100
d_model = 256
softmax = nn.Softmax(dim = 1)

def Softmax(Z):
    Zmax, _ = torch.max(Z, dim = -1, keepdim=True)
    Z_exp = torch.exp(Z - Zmax)
    return Z_exp / torch.sum(Z_exp, dim = 1, keepdims = True)

# Self attension

"""
X(Seq, d_model)
W_qkv(d_model, 3*d_model)
Split W_qkv(d_model, 3*d_model) into W_q, W_k, W_v(d_model, d_model)
Q, K, V(Seq, d_model)
Result(Seq, d_model)
"""

def Self_attension(X, W_qkv, d_model, W_o):
    W_q, W_k, W_v = torch.split(W_qkv, d_model, dim=0)
    Q = F.linear(X, W_q)
    K = F.linear(X, W_k)
    V = F.linear(X, W_v)
    attention_scores = torch.matmul(Q, K.T) / sqrt(d_model)
    attention_weights = softmax(attention_scores)
    result = torch.matmul(attention_weights, V)
    result = torch.matmul(result, W_o)
    return result

# Positional encoding

"""
X(Seq, d_model)
encoding(Seq, d_model)
position(Seq, 1)
div(d_model / 2, 1)
X_embedding(Seq, d_model)
"""

def Positional_encoding(X):
    encoding = torch.zeros(Seq, d_model)
    position = torch.arange(0, Seq).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div)
    return X + encoding

def Layer_norm(X):
    layer_norm = nn.LayerNorm(X.size()[1:])
    output = layer_norm(X)
    return output

# Transformer

"""
Input dim(Seq, d_model)
Output dim(Seq, d_model)
"""

class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_encoder = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        self.W_qkv = self.attn_encoder.in_proj_weight
        self.W_o = self.attn_encoder.out_proj.weight
        self.Mask = torch.triu(-float('inf') * torch.ones(Seq, Seq), 1)
    def forward(self, x):
        x_PE = Positional_encoding(x)
        x_norm = Layer_norm(x)
        result = Self_attension(x_norm, self.W_qkv, d_model, self.W_o) + x_PE             # Resnet
        
        test1 = Self_attension(x_norm, self.W_qkv, d_model, self.W_o)
        test2, _ = self.attn_encoder(x_norm, x_norm, x_norm)
        print(np.linalg.norm(test1.detach().numpy() - test2.detach().numpy()))

        plt.imshow(result.detach().numpy(), cmap='viridis')
        plt.show()
        return result

# Test

X = torch.randn(Seq, d_model)
model = Transformer()
result = model(X)
