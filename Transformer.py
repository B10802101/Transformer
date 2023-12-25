import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt

# Multihead attention

"""
Assume q, k, v dim(batchsize, seq_length, d_model)
mask = True/False
"""
class Multihead_attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Multihead_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        # For optimization, updating WQKV parameters

        self.LTwq = nn.Linear(d_model, d_model)     
        self.LTwk = nn.Linear(d_model, d_model)
        self.LTwv = nn.Linear(d_model, d_model)
        self.LTwo = nn.Linear(d_model, d_model)

    # Scaled_dot_product_attention

    """
    X(Seq, d_model)
    q, k, v(batch_size, num_heads, seq_length, depth)
    attention_weights(d_model, d_model)
    result(batch_size, num_heads, seq_length, depth)
    """

    def Scaled_dot_product_attention(self, q, k, v, Mask):
        d_model = k.shape[-1]
        k_t = k.permute(0, 1, 3, 2)
        attention_scores = torch.matmul(q, k_t) / (d_model ** 0.5)
        if Mask == True:
            Lookahead_mask = self.lookahead_mask(k.shape[-2])
            attention_scores += Lookahead_mask * -1e9
        attention_weights = F.softmax(attention_scores, dim=-1)
        result = torch.matmul(attention_weights, v)
        return attention_weights, result

    # Creating a lookahead Mask

    """
    For Decoder
    0 1 1 1 1 1 1 1 1...
    0 0 1 1 1 1 1 1 1...
    0 0 0 1 1 1 1 1 1...
    0 0 0 0 1 1 1 1 1...
    0 0 0 0 0 1 1 1 1...
    ...
    """

    def lookahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size)) - torch.eye(size)
        return mask

    # Split heads

    """
    Multihead used to put attention on different subwords
    d_model = number of heads * depth, that is, the reprs of a subword are divided into many heads with less repr
    input dim(batch_size, seq_length, d_model)
    output dim(batch_size, num_heads, seq_length, depth)
    """

    def split_heads(self, x, num_heads):
        batch_size = x.shape[0]
        reshaped_x = x.reshape(batch_size, -1, num_heads, self.depth)    # reshaped_x dim(batch_size, seq_length, num_heads, depth)
        output = reshaped_x.permute(0, 2, 1, 3)     # output dim(batch_size, num_heads, seq_length, depth)
        return output
    
    # Forward

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]
        seq_length = q.shape[1]

        q = self.LTwq(q)
        k = self.LTwk(k)
        v = self.LTwv(v)

        q = self.split_heads(q, self.num_heads)     # (batch_size, num_heads, seq_length, depth)
        k = self.split_heads(k, self.num_heads)     # (batch_size, num_heads, seq_length, depth)
        v = self.split_heads(v, self.num_heads)     # (batch_size, num_heads, seq_length, depth)

        att_w, result = self.Scaled_dot_product_attention(q, k, v, mask)    # result (batch_size, num_heads, seq_length, depth)
        result = result.permute(0, 2, 1, 3)    # result (batch_size, seq_length, num_heads, depth)
        stack_result = result.reshape(batch_size, seq_length, -1)    # result (batch_size, seq_length, d_model)
        
        output = self.LTwo(stack_result)

        return output, att_w

# FFN

"""
output dim(batch_size, seq_length, d_model)
p.s. The same linear transform to all position of the subwords
"""
class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(PositionwiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Encoderlayer

"""
mask = True/False
model.eval() turned dropout off
output dim(batch_size, seq_length, d_model)
"""

class Encoderlayer(nn.Module):
    def __init__(self, x, d_model, num_heads, dff, rate=0.1):
        super(Encoderlayer, self).__init__()
        self.mha = Multihead_attention(d_model, num_heads)
        self.ffn = PositionwiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
    
    # forward

    def forward(self, x):
        attn_output, attnw = self.mha(x, x, x, False)
        attn_output = self.dropout1(attn_output)
        x = self.layernorm1(attn_output + x)

        x_out1 = self.ffn(x)
        x_out1 = self.dropout2(x_out1)
        x_out2 = self.layernorm2(x + x_out1)

        return x_out2

# Decoderlayer

"""
All sublayer's output dim(batch_size, seq_length, d_model)
enc_out dim(batch_size, seq_length, d_model)
mha1(q, k, v, mask)
mha2(q, k, v, mask)
model.eval() turned dropout off
"""

class Decoderlayer(nn.Module):
    def __init__(self, x, d_model, num_heads, dff, rate=0.1):
        super(Decoderlayer, self).__init__()
        self.mha1 = Multihead_attention(d_model, num_heads)
        self.mha2 = Multihead_attention(d_model, num_heads)
        self.ffn = PositionwiseFeedForwardNetwork(d_model, dff)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)
    def forward(self, x, mask, enc_out):
        attn1, _ = self.mha1(x, x, x, mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(out1, enc_out, enc_out, mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        out3 = self.ffn(out2)
        out3 = self.dropout3(out3)
        out3 = self.layernorm3(out2 + out3)

        return out3

# Positional encoding

"""
X(batch_size, Seq, d_model)
encoding(batch_size, Seq, d_model)
position(batch_size, Seq, 1)
div(d_model / 2)
X_embedding(batch_size, Seq, d_model)
"""

def Positional_encoding(batch_size, Seq, d_model):
    encoding = torch.zeros(batch_size, Seq, d_model)
    position = torch.arange(0, Seq).unsqueeze(0).repeat(batch_size, 1).float().unsqueeze(2)    # (batch_size, Seq, 1)
    print(position)
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    encoding[:, :,  0::2] = torch.sin(position * div)
    encoding[:, :, 1::2] = torch.cos(position * div)
    return encoding

# Encoder

"""
Input dim (batch_size, Seq_length)
Output dim (batch_size, Seq_length, d_model)
Encoder consists of Encoderlayers
Additional param: num_layers, vocab_size
vocab_size is the amounts of vocabulary
Positional_encoding encode all inputs(batch_size, Seq_length, d_model)
Encoder has no mask
Embedding is also trainable param
"""

class Encoder(nn.Module):
    def __init__(self, x, d_model, num_heads, dff, num_layers, vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = Positional_encoding(x.shape[0], x.shape[1], d_model)

        self.enc_layers = [Encoderlayer(x, d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = nn.Dropout(rate)
    def forward(self, x):
        x = self.embedding(x.to(torch.long))   # (batch_size, seq_length, d_model)
        x *= self.d_model ** 0.5
        x += self.pos_encoding
        x = self.dropout(x)

        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x)
        
        return x

# Decoder

"""
Input dim (batch_size, seq_length)
Output dimn (batch_size, seq_length, d_model)
"""
class Decoder(nn.Module):
    def __init__(self, x, d_model, num_heads, dff, num_layers, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = Positional_encoding(x.shape[0], x.shape[1], d_model)

        self.dec_layers = [Decoderlayer(x, d_model, num_heads, dff, rate)
                           for _ in range(num_heads)]
        self.dropout = nn.Dropout(rate)
    def forward(self, x, mask, enc_out):
        x = self.embedding(x.to(torch.long))    # (batch_size, seq_length, d_model)
        x *= self.d_model**0.5
        x += self.pos_encoding
        x = self.dropout(x)

        for i, dec_layer in enumerate(self.dec_layers):
            x = dec_layer(x, mask, enc_out)

        return x



# # Transformer

# """
# Input dim(Seq, d_model)
# Attention weights dim(d_model, d_model), sum of rows is 1
# Output dim(Seq, d_model)
# """

# class Transformer(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.attn_encoder = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
#         self.W_qkv = self.attn_encoder.in_proj_weight
#         self.W_o = self.attn_encoder.out_proj.weight
#         self.Mask = torch.triu(-float('inf') * torch.ones(Seq, Seq), 1)
#     def forward(self, x):
#         x_PE = Positional_encoding(x)
#         x_norm = Layer_norm(x)
#         # result = Self_attension(x_norm, self.W_qkv, d_model, self.W_o)[1] + x_PE             # Resnet
        
#         att_W, test1 = Self_attension(x_norm, self.W_qkv, d_model, self.W_o, True)
#         test2, _ = self.attn_encoder(x_norm, x_norm, x_norm)
#         print(np.linalg.norm(test1.detach().numpy() - test2.detach().numpy()))

#         # plt.imshow(result.detach().numpy(), cmap='viridis')
#         # plt.show()
#         # return result

# Test
heads = 2
d_model = 4
batch_size = 10
dff = 8
Seq = 100
vocab_size = 2000
x = torch.randint(0, vocab_size, (batch_size, Seq))
enc_out = torch.randn(batch_size, Seq, d_model)
dec = Decoder(x, d_model, heads, dff, 2, vocab_size, 0.1)
out = dec(x, True, enc_out)
print(f'out shape: {out.shape}')
print(f'out: {out}')
