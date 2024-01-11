import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias = in_proj_bias) # We multiplied by 3 as we need to make key, query and value matrices
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias) # each pixel is a token that has its own embedding represented by he number of channels
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, causal_mask = False) -> torch.Tensor:
        # x: (Batch size, Sequence length, Embedding dimension)
        
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        
        # (Batch size, Sequence length, Embedding dimension) -> 3 tensors of shape (Batch size, Sequence length, Embedding dimension)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)
        
        # (Batch size, Sequence length, Embedding dimension) -> (Batch size, Sequence length, Heads, Dim/Heads) -> (Batch size, Heads, Sequence length, Dim/Heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        # (Batch size, Heads, Sequence length, Dim/Heads) -> (Batch size, Heads, Sequence length, Sequence length)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the prncipal diagonal) is made up of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu_(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim = -1)
        
        # (Batch size, Heads, Sequence length, Sequence length) @ (Batch Size, Heads, Seq_Len, Dim/Heads) -> (Batch size, Heads, Sequence length, Dim/Heads)
        output = weight @ v
        
        # (Batch size, Heads, Seq_Len, Dim/Heads) -> (Batch Size, Seq_Len, Heads, Dim/Heads)
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        # (Batch size, Sequence length, Embedding dimension)
        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_cross, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (latent): (Batch size, Sequence length, Dim Q)
        # y: (context): (Batch size, Sequence length, Dim KV) = (Batch size, 77, 768)
        
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # Multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim = -1)
        
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        
        output = self.out_proj(output)
        
        return output