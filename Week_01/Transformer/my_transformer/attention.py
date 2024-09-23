import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO

        ## (kyu) q, k, v 차원: (batch size, vocab_size, d_model)

        attention_score = torch.matmul(q, k.transpose(-2, -1))
        attention_score = attention_score / math.sqrt(q.shape[-1])
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_prob = F.softmax(attention_score, dim = -1)   ## (kyu) (batch size, vocab_size, vocab_size)
        out = torch.matmul(attention_prob, v)

        return out, attention_prob

class MultiHeadAttention(nn.Module):   ## (kyu) scaled dot product attention을 병렬적으로 수행하는 layer, linear projection -> 병렬 attention -> concatenate -> linear projection
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)   ## (kyu) 마지막 선형 변환
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        
        batch_size = Q.size(0)

        Q = self.query_layers(Q).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        K = self.key_layers(K).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        V = self.value_layers(V).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)

        out, scaled_attention = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)

        return self.fc(out)