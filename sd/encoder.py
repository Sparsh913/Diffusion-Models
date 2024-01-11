import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch size, Channel, Height, Width) -> (Batch size, 128, Height, Width)
            nn.Conv2d(3,128, kernel_size=3, padding=1),
            
            # (Batch size, 128, Height, Width) -> (Batch size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch size, 128, Height, Width) -> (Batch size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch size, 128, Height, Width) -> (Batch size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch size, 128, Height/2, Width/2) -> (Batch size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),
            
            # (Batch size, 256, Height/2, Width/2) -> (Batch size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),
            
            # (Batch size, 256, Height/2, Width/2) -> (Batch size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            # (Batch size, 256, Height/4, Width/4) -> (Batch size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),
            
            # (Batch size, 512, Height/4, Width/4) -> (Batch size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            
            # (Batch size, 512, Height/4, Width/4) -> (Batch size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 512, Height/8, Width/8)
            nn.SiLU(),
            
            # (Batch size, 512, Height/8, Width/8) -> (Batch size, 8, Height/8, Width/8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # (Batch size, 8, Height/8, Width/8) -> (Batch size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), # Why 1x1 conv?
            
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Channel, Height, Width)
        # noise: (Batch size, Out_Channel, Height/8, Width/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                x = F.pad(x, (0,1,0,1))
            x = module(x)
        
        # (Batch size, 8, Height/8, Width/8) -> two tensors of shape (Batch_size, 4, Height/8, Width/8)    
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # (Batch size, 4, Height/8, Width/8) -> (Batch size, 4, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        
        # (Batch size, 4, Height/8, Width/8) -> (Batch size, 4, Height/8, Width/8)
        variance = log_variance.exp()
        
        # (Batch size, 4, Height/8, Width/8) -> (Batch size, 4, Height/8, Width/8)
        stdev = variance.sqrt()
        
        # Z = N(0,1) -> N(mean, variance) = X
        # X = mean + stdev * Z
        x = mean + stdev * noise
        
        # Scale the output by a constant
        x = x * 0.18215
        
        return x