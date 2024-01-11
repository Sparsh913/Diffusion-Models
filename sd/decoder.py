import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.attention = SelfAttention(1, in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Batch_Size, Features, Height, Width)
        
        residue = x
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h * w)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view(n, c, h, w)
        
        x = x + residue
        
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x):
        # x: Batch_Size, In_channels, Height, Width)
        
        residue = x
        
        x = self.group_norm1(x)
        
        x = F.silu(x)
        
        x = self.conv1(x)
        
        x = self.group_norm2(x)
        
        x = F.silu(x)
        
        x = self.conv2(x)
        
        x = x + self.residual_layer(residue)
        
        return x
    
class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4, kernel_size=1, padding=0),
            
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/4, Width/4)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batch size, 512, Height/4, Width/4) -> (Batch size, 256, Height/2, Width/2)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            # (Batch size, 256, Height/2, Width/2) -> (Batch size, 256, Height, Width)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128),
            
            nn.SiLU(),
            
            # (Batch size, 128, Height, Width) -> (Batch size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, 4, Height/8, Width/8)
        x /= 0.18215
        
        for module in self:
            x = module(x)
            
        # x: (Batch size, 3, Height, Width)
        return x