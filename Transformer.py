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
    attention_weights(batch_size, num_heads, seq_length, seq_length)
    result(batch_size, num_heads, seq_length, depth)
    """

    def Scaled_dot_product_attention(self, x, q, k, v, Mask):
        d_model = k.shape[-1]
        k_t = k.permute(0, 1, 3, 2)
        attention_scores = torch.matmul(q, k_t) / (d_model ** 0.5)
        if Mask is not None:
            attention_scores += Mask * -1e9
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

    def lookahead_mask(self, seq_length_q, seq_length_k):
        lookahead_mask = torch.triu(torch.ones(seq_length_q, seq_length_k)) - torch.eye(seq_length_q, seq_length_k)
        return lookahead_mask

    # Creating a padding mask
    
    def create_padding_mask(self, seq):
        padding_mask = torch.eq(seq, 0).float()
        return padding_mask.unsqueeze(1).unsqueeze(1)

    def create_lookahead_mask(self, size):
        lookahead_mask = torch.triu(torch.ones(size, size)) - torch.eye(size, size)
        return lookahead_mask
    
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

    def forward(self, x, q, k, v, mask):
        batch_size = q.shape[0]
        seq_length = q.shape[1]

        q = self.LTwq(q)
        k = self.LTwk(k)
        v = self.LTwv(v)

        q = self.split_heads(q, self.num_heads)     # (batch_size, num_heads, seq_length, depth)
        k = self.split_heads(k, self.num_heads)     # (batch_size, num_heads, seq_length, depth)
        v = self.split_heads(v, self.num_heads)     # (batch_size, num_heads, seq_length, depth)

        att_w, result = self.Scaled_dot_product_attention(x, q, k, v, mask)    # result (batch_size, num_heads, seq_length, depth)
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
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(Encoderlayer, self).__init__()
        self.mha = Multihead_attention(d_model, num_heads)
        self.ffn = PositionwiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
    
    # forward

    def forward(self, x, mask):
        attn_output, attnw = self.mha(x, x, x, x, mask)
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
    def __init__(self, d_model, num_heads, dff, rate=0.1):
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
    def forward(self, x, combine_mask, padding_mask, enc_out):
        attn1, attn1_w = self.mha1(x, x, x, x, combine_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        # enc_out = enc_out[:, :x.shape[1], :]

        attn2, attn2_w = self.mha2(out1, out1, enc_out, enc_out, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        out3 = self.ffn(out2)
        out3 = self.dropout3(out3)
        out3 = self.layernorm3(out2 + out3)

        return out3, attn1_w, attn2_w

# Positional encoding

"""
X(batch_size, Seq, d_model)
encoding(batch_size, Seq, d_model)
position(batch_size, Seq, 1)
div(d_model / 2)
X_embedding(batch_size, Seq, d_model)
"""

def Positional_encoding(batch_size, seq_length, d_model):
    encoding = torch.zeros(batch_size, seq_length, d_model)
    position = torch.arange(0, seq_length).unsqueeze(0).repeat(batch_size, 1).float().unsqueeze(2)    # (batch_size, Seq, 1)
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

        self.enc_layers = [Encoderlayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = nn.Dropout(rate)
    def forward(self, x, mask):
        x = self.embedding(x.to(torch.long))   # (batch_size, seq_length, d_model)
        x *= self.d_model ** 0.5
        x += self.pos_encoding
        x = self.dropout(x)

        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, mask)
        
        return x

# Decoder

"""
Input dim (batch_size, seq_length)
Output dimn (batch_size, seq_length, d_model)
return output, attention weights history
"""
class Decoder(nn.Module):
    def __init__(self, tar, d_model, num_heads, dff, num_layers, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = Positional_encoding(tar.shape[0], tar.shape[1]-1, d_model)    # Throwing seq-1 into Decoder

        self.dec_layers = [Decoderlayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = nn.Dropout(rate)
    def forward(self, tar, combine_mask, padding_mask, enc_out):
        attention_weights = {}

        tar = self.embedding(tar.to(torch.long))    # (batch_size, seq_length, d_model)
        tar *= self.d_model**0.5
        tar += self.pos_encoding
        tar = self.dropout(tar)

        for i, dec_layer in enumerate(self.dec_layers):
            tar, attn1_w, attn2_w = dec_layer(tar, combine_mask, padding_mask, enc_out)

            attention_weights[f'decoderlayer{i+1} attention weights 1'] = attn1_w
            attention_weights[f'decoderlayer{i+1} attention weights 2'] = attn2_w
        return tar, attention_weights



# Transformer
    
"""
Transformer = Encoder + Decoder + Final linear layer
forward input (x, tar)
output (batch_size, seq_length, target_vocab_size)

"""

class Transformer(nn.Module):
    def __init__(self,  x, tar, d_model, num_heads, dff, 
                 num_layers, input_vocab_size, target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(x, d_model, num_heads, dff, num_layers, input_vocab_size, rate)    # (batch_size, seq_length, d_model)
        self.decoder = Decoder(tar, d_model, num_heads, dff, num_layers, target_vocab_size, rate)    # (batch_size, seq_length, d_model)
        self.final_ffn = nn.Linear(d_model, target_vocab_size)    # (batch_size, seq_length, target_vocab_size)
    
    # forward
    def forward(self, x, tar, input_mask, combine_mask, padding_mask):
        enc_out = self.encoder(x, input_mask)
        dec_out, attention_weight_history = self.decoder(tar, combine_mask, padding_mask, enc_out)
        final_output = self.final_ffn(dec_out)
        
        return final_output, attention_weight_history
    
# Padding mask & lookahead mask
    
def create_padding_mask(seq):
    padding_mask = torch.eq(seq, 0).float()
    return padding_mask.unsqueeze(1).unsqueeze(1)

def create_lookahead_mask(size):
        lookahead_mask = torch.triu(torch.ones(size, size)) - torch.eye(size, size)
        return lookahead_mask

# Test
heads = 2
d_model = 4
batch_size = 2
dff = 8
Seq = 8
num_layers = 1
inp_vocab_size = 20
tar_vocab_size = 13

tar_input = torch.randint(0, tar_vocab_size, (batch_size, Seq))
input = torch.randint(0, inp_vocab_size, (batch_size, Seq))
# input = torch.Tensor([[1, 1, 1, 1, 1, 1, 0, 0],[1,1,1,1,1,0,1,1]])

input_mask = create_padding_mask(input)
tar_padding_mask = create_padding_mask(tar_input[:, :-1])
tar_lookahead_mask = create_lookahead_mask(tar_input[:, :-1].shape[1])
combine_mask = torch.maximum(tar_padding_mask, tar_lookahead_mask)
tar_padding_mask = tar_padding_mask.squeeze(1).unsqueeze(3)


model = Transformer(input, tar_input, d_model, heads, dff, num_layers, inp_vocab_size, tar_vocab_size, 0.1)  
out, attnw = model(input, tar_input[:, :-1], input_mask, combine_mask, tar_padding_mask)
print(f'out shape: {out.shape}')
print(f'out: {out}')
print(f'attnw: {attnw}')