import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import trunc_normal_

class EmbeddingLayer(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, batch_size: int = 128) -> None:
        super(EmbeddingLayer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim   # (kyu) patch 1개당 length가 몇인 vector로 표현이 되냐

        self.project = nn.Conv2d(in_channels, self.embed_dim, kernel_size = patch_size, stride = patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim))

    def forward(self, x) :
        x = x.permute(0, 3, 1, 2)
        x = self.project(x).flatten(2).transpose(1, 2)
        repeated_cls = self.cls_token.repeat(x.size()[0], 1, 1)
        x = torch.cat((repeated_cls, x), dim = 1)
        x += self.pos_emb

        return x
        
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, N, embed_dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.attn_dropout(attn_prob)
        
        output = torch.matmul(attn_prob, v).transpose(1, 2).contiguous().reshape(batch_size, N, embed_dim)
        
        output = self.proj(output)
        output = self.proj_dropout(output)
        
        return output


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout: float = 0.1) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout = 0.1) -> None:
        super(Block, self).__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout = dropout)
        self.LN = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)

    def forward(self, x) -> torch.Tensor:
        x = x + self.attn(self.LN(x))
        x = x + self.mlp(self.LN(x))

        return x

    #여기까지 Encoder 구현 끝!!
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size = 65, patch_size = 5, num_heads = 4, depth = 8, embed_dim = 32, hidden_dim = 64, in_channels = 1, num_classes = 3, dropout = 0.1) -> None:
        super(VisionTransformer, self).__init__()
        
        self.embedding = EmbeddingLayer(img_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.final = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes))


    def forward(self, x):
        x = self.embedding(x)
    
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim = 1)
        x = self.final(x)
        
        return x