import torch
import torch.nn.functional as F

def self_attention(X, W_qkv, d_model, W_o):
    # 分割权重矩阵
    W_q, W_k, W_v = torch.split(W_qkv, d_model, dim=0)
    
    # 计算 Q、K、V
    Q = F.linear(X, W_q)
    K = F.linear(X, W_k)
    V = F.linear(X, W_v)
    
    # 缩放点积注意力
    attention_scores = torch.matmul(Q, K.T) / (d_model ** 0.5)
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    # 计算加权和
    result = torch.matmul(attention_weights, V)
    
    # 线性变换得到最终输出
    result = torch.matmul(result, W_o.T)
    
    return result

# 示例输入
d_model = 256
sequence_length = 100
X = torch.randn(sequence_length, d_model)

# 使用 nn.MultiheadAttention 进行计算
attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=1, bias=False)
W_qkv = attn.in_proj_weight
W_o = attn.out_proj.weight
result_builtin, _ = attn(X, X, X)

# 使用自定义函数进行计算
result_custom = self_attention(X, W_qkv, d_model, W_o)

# 计算结果差异的 L2 范数
norm = torch.norm(result_custom - result_builtin)
print(norm.item())
