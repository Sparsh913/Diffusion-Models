import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super.__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))
        
    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # (Batch Size, Seq_Len) -> (Batch Size, Seq_Len, Dimension)
        x = self.token_embedding(tokens)
        
        x += self.position_embedding
        
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch Size, Seq_Len, Dimension)
        residue = x
        
        # Self Attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal=True)
        x += residue
        
        # Feed Forward
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702*x) # QuickGELU Activation Function
        
        x += residue

        return x
        
        
class CLIP(nn.Module):
    
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
                                 ])
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        tokens = tokens.type(torch.long)
        
        # (Batch Size, Seq_Len) -> (Batch Size, Seq_Len, Dimension)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        # (Batch Size, Seq_Len, Dimension)    
        output = self.layernorm(state)
        
        return output