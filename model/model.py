import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ModelConfig
class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        assert config.d_model % config.n_heads == 0, "Number of dim should be divisible by the n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.dropout = config.attn_dropout

        self.d_k = self.d_model / self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.qkv_bias)

        self.out_proj = nn.Linear(self.n_heads * self.d_k, self.d_model, bias=config.qkv_bias)

    def forward(self, x):
        B, T, _ = x.shape

        query = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2) # B, N_H, T, D_K
        key = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        value = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # TODO: replace with flash attention for cuda
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) // math.sqrt(self.d_k)
        attn_scores = F.softmax(attn_scores) #B, N_H, T, T
        attn_scores = torch.matmul(value, attn_scores).view(B, T, self.n_heads * self.d_k).contiguous()
        # TODO: Find out where dropout fits in

        out = self.out_proj(attn_scores) # B, T, C
        return out
    









