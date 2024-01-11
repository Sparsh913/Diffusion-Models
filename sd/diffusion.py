import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module): # To return the time step of the noise
    def __init__(self, n_embed: int):
        super().__init__()
        
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        
        # (1, 1280)
        return x
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity() # connect directly in_channels and out_channels directly with the residual connections
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, feature, time): # This is to relate the latent space to the time embedding
        # feature: (Batch size, In_Channels, Height, Width)
        # time: (1, 1280)
        
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_context = 768):
        super().__init__()
        channels = n_heads * n_embed # How?
        self.groupnorm = nn.GroupNorm(32, channels, eps = 1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layernorm1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layernorm3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, channels * 4 * 2)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, features, Height, Width)
        # context: (Batch size, Seq_Len, Dimension)
        
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch size, features, Height, Width) -> (Batch size, features, Height * Width)
        x = x.view(n, c, h * w)
        # (Batch size, features, Height * Width) -> (Batch size, Height * Width, features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connnections
        residue_short = x
        
        x = self.layernorm1(x)
        x = self.attention_1(x)
        x += residue_short
        
        residue_short = x
        
        # Normalization + Cross-Attention with skip connections
    
        x = self.layernorm2(x)
        
        # Cross Attention
        x = self.attention_2(x, context)
        x += residue_short
        
        residue_short = x
        
        # Normalization + FF with GeGLU and skip connections
        x = self.layernorm3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim = -1)
        x = x * F.gelu(gate)
        
        x = self.linear_geglu_2(x)
        x += residue_short
        
        # (Batch size, Height * Width, features) -> (Batch size, features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch size, features, Height * Width) -> (Batch size, features, Height, Width)
        x = x.view(n, c, h, w)
        
        return self.conv_output(x) + residue_long
    
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (Batch size, Channels, Height, Width) -> (Batch size, Channels, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class SwitchSequential(nn.Sequential):
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, time = torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
        
    
class UNET(nn.Module): # Is it just predicting how much noise there is?
    def __init__(self):
        super().__init__()
        
        self.encoders = nn.Module([
            
            # (Batch size, 4, Height/8, Width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch size, 320, Height/8, Width/8) -> (Batch size, 320, Height/16, Width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch size, 640, Height/16, Width/16) -> (Batch size, 640, Height/32, Width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch size, 1280, Height/32, Width/32) -> (Batch size, 1280, Height/64, Width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch size, 1280, Height/64, Width/64) -> (Batch size, 1280, Height/64, Width/64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),   
        )
        
        self.decoders = nn.ModuleList([
            # (Batch size, 1280, Height/64, Width/64) -> (Batch size, 1280, Height/64, Width/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), # Due to the skip connections, the input expected is double the output of the previous layer, hence 2560
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), Upsample(1280), 
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), UNET_AttentionBlock(8, 160),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), UNET_AttentionBlock(8, 160),
            SwitchSequential(UNET_ResidualBlock(1920, 1280)), UNET_AttentionBlock(8, 160), Upsample(1280),
            SwitchSequential(UNET_ResidualBlock(1920, 640)), UNET_AttentionBlock(8, 80),
            SwitchSequential(UNET_ResidualBlock(1280, 640)), UNET_AttentionBlock(8, 80), 
            SwitchSequential(UNET_ResidualBlock(960, 640)), UNET_AttentionBlock(8, 80), Upsample(640),
            SwitchSequential(UNET_ResidualBlock(960, 320)), UNET_AttentionBlock(8, 40),
            SwitchSequential(UNET_ResidualBlock(640, 320)), UNET_AttentionBlock(8, 40),
            SwitchSequential(UNET_ResidualBlock(640, 320)), UNET_AttentionBlock(8, 40)
        ])
        
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: (Batch size, Channels, Height/8, Width/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        
        # (Batch size, 4, Height/8, Width/8)
        return x
    
class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
        
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch Size, 4, Height/8, Width/8)
        # context: (Batch Size, Seq_Len, Dimension) Context is the prompt
        # time: (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch Size, 4, Height/8, Width/8) -> (Batch Size, 320, Height/8, Width/8)
        output = self.unet(latent, context, time)
        
        # (Batch Size, 320, Height/8, Width/8) -> (Batch Size, 4, Height/8, Width/8)
        output = self.final(output)
        
        # (Batch Size, 4, Height/8, Width/8)
        return output