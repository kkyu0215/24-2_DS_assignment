import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):   ## (kyu) 룩업 테이블 생성
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)   ## (kyu) 1개의 단어가 d_model 차원의 vector로 embedding
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):   ## (kyu) 위치 정보 나타내는 positional embedding 생성하여 token embedding에 더해준 후 encoder, decoder에 넣어주기
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #TODO
        pos_embedding = torch.zeros((max_len, d_model))
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)   ## (kyu) update 되지 않도록

    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.pos_embedding[:x.size(-2), :]