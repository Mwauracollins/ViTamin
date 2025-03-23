import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Resize, ToTensor
from torchvision.datasets import OxfordIIITPet

from config.model_config import ModelConfig

from einops import repeat, rearrange

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "Number of dim should be divisible by the n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.dropout = config.att_dropout

        self.d_k = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.qkv_bias)

        self.out_proj = nn.Linear(self.n_heads * self.d_k, self.d_model, bias=config.qkv_bias)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape

        query = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2) # B, N_H, T, D_K
        key = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        value = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # TODO: replace with flash attention for cuda
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn_scores = F.softmax(attn_scores, dim=-1) #B, N_H, T, T
        attn_scores = self.attn_dropout(attn_scores)

        attn_output = torch.matmul(attn_scores, value)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.n_heads * self.d_k)

        out = self.out_proj(attn_output) # B, T, C
        out = self.proj_dropout(out)
        return out
    

class FeedForward(nn.Sequential):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.p_proj = nn.Linear(patch_size * patch_size * in_channels, d_model)

    def forward(self, x: torch.Tensor):
        
        # Using einops instead of unfold
        patches = rearrange(
            x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
            p1=self.patch_size, p2=self.patch_size
        )

        out = self.p_proj(patches)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.attention = Attention(config=config)
        self.layernorm2 = nn.LayerNorm(config.d_model)
        self.feedforward = FeedForward(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            dropout=config.att_dropout
        )
    
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.feedforward(self.layernorm2(x))
        return x
    

class ViT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.in_channels = config.n_channels
        self.height = config.image_size
        self.width = config.image_size
        self.patch_size = config.patch_size
        self.n_layers = config.n_layers

        # Patching
        self.patch_emb = PatchEmbedding(config.n_channels, config.patch_size, config.d_model)

        # Position embedding
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, config.d_model))
        self.cls_token = nn.Parameter(torch.rand(1, 1, config.d_model))

        # Attention blocks
        self.blocks = nn.ModuleList([AttentionBlock(self.config) for _ in range(self.n_layers)])
        
        # classification head
        self.layernorm_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.out_dim)

    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)

        # add CLS token to inputs
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat([cls_token, x], dim=1)

        # add pos embedding
        x = x + self.pos_emb

        for block in self.blocks:
            x = block(x)
        
        x = self.layernorm_f(x[:, 0])
        out = self.head(x)

        return out