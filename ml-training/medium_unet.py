#!/usr/bin/env python3
"""
Medium U-Net architecture for patch-based denoising from 128x128 inputs.

Input: 128x128x3 RGB patches (noisy images)
Output: 32x32x3 RGB patches (center region prediction) + 4 transform parameters

This is a more sophisticated version of CompactUNet with:
- Deeper network (5 encoder levels)
- More channels at each level
- Residual blocks with GroupNorm
- Attention gates in decoder
- Better transform detection head

Estimated parameters: ~9-10M (vs ~1.35M for CompactUNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GroupNormAct(nn.Module):
    """GroupNorm + Activation block."""
    
    def __init__(self, num_channels: int, num_groups: int = 8, activation: str = 'silu'):
        super(GroupNormAct, self).__init__()
        self.norm = nn.GroupNorm(min(num_groups, num_channels), num_channels)
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        return self.act(self.norm(x))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions, GroupNorm, and skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm_act1 = GroupNormAct(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm_act2 = GroupNormAct(out_channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm_act1(out)
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm_act2(out)
        
        return out + residual


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features during skip connections."""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int = None):
        super(AttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
        
        self.W_gate = nn.Conv2d(gate_channels, inter_channels, 1, stride=1, padding=0)
        self.W_skip = nn.Conv2d(skip_channels, inter_channels, 1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, gate, skip):
        g1 = self.W_gate(gate)
        x1 = self.W_skip(skip)
        
        # Upsample gate signal if needed
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.sigmoid(psi)
        
        return skip * psi


class EncoderBlock(nn.Module):
    """Enhanced encoder block with residual connections and optional SE attention."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, use_se: bool = True):
        super(EncoderBlock, self).__init__()
        
        self.res_block1 = ResidualBlock(in_channels, out_channels, dropout)
        self.res_block2 = ResidualBlock(out_channels, out_channels, dropout)
        
        self.se = SEBlock(out_channels) if use_se else None
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        if self.se is not None:
            x = self.se(x)
        
        skip = x
        x = self.pool(x)
        
        return x, skip


class DecoderBlock(nn.Module):
    """Enhanced decoder block with attention gates and residual connections."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, 
                 dropout: float = 0.0, use_attention: bool = True, use_transpose: bool = True):
        super(DecoderBlock, self).__init__()
        
        self.use_transpose = use_transpose
        self.use_attention = use_attention
        
        # Upsampling
        if use_transpose:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.upsample_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        
        # Attention gate
        if use_attention:
            self.attention = AttentionGate(in_channels // 2, skip_channels)
        
        # Concatenated channels: upsampled + (attended) skip connection
        concat_channels = (in_channels // 2) + skip_channels
        
        self.res_block1 = ResidualBlock(concat_channels, out_channels, dropout)
        self.res_block2 = ResidualBlock(out_channels, out_channels, dropout)
    
    def forward(self, x, skip):
        # Upsample
        if self.use_transpose:
            x = self.upsample(x)
        else:
            x = self.upsample(x)
            x = self.upsample_conv(x)
        
        # Handle size mismatch
        if x.size() != skip.size():
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        
        # Apply attention to skip connection
        if self.use_attention:
            skip = self.attention(x, skip)
        
        # Concatenate and process
        x = torch.cat([x, skip], dim=1)
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        return x


class MediumUNet(nn.Module):
    """
    Medium U-Net for center patch prediction from 128x128 patches.
    
    Architecture:
    - Encoder: 128x128 → 64x64 → 32x32 → 16x16 → 8x8 → 4x4
    - Bottleneck: 4x4 with 512 channels and residual blocks
    - Decoder: 4x4 → 8x8 → 16x16 → 32x32 (exact 32x32 output)
    - Transform head: Global features → 4 scalars
    
    Estimated parameters: ~9-10M
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid', use_attention: bool = True):
        super(MediumUNet, self).__init__()
        
        self.final_activation = final_activation
        
        # Encoder path - 4 levels (reduced from 5 to control parameters)
        # 128x128 → 64x64
        self.enc1 = EncoderBlock(in_channels, 32, dropout * 0.5)
        # 64x64 → 32x32  
        self.enc2 = EncoderBlock(32, 64, dropout * 0.7)
        # 32x32 → 16x16
        self.enc3 = EncoderBlock(64, 128, dropout * 0.8)
        # 16x16 → 8x8
        self.enc4 = EncoderBlock(128, 256, dropout)
        
        # Bottleneck at 8x8 with 256 channels
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 256, dropout),
            ResidualBlock(256, 256, dropout),
            SEBlock(256)
        )
        
        # Decoder path - to exact 32x32 output
        # 8x8 → 16x16 (with skip from enc4: 256 channels)
        self.dec4 = DecoderBlock(256, 256, 128, dropout, use_attention, use_transpose)
        # 16x16 → 32x32 (with skip from enc3: 128 channels)
        self.dec3 = DecoderBlock(128, 128, 64, dropout, use_attention, use_transpose)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            GroupNormAct(32),
            nn.Conv2d(32, out_channels, 1)
        )
        
        # Transform prediction head - more sophisticated than CompactUNet
        self.transform_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 256, 1),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(256, 64, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 4, 1),
            nn.Flatten()
        )
    
    def forward(self, x):
        # Encoder path
        x1, skip1 = self.enc1(x)    # skip1: 128x128x32, x1: 64x64x32
        x2, skip2 = self.enc2(x1)   # skip2: 64x64x64, x2: 32x32x64
        x3, skip3 = self.enc3(x2)   # skip3: 32x32x128, x3: 16x16x128
        x4, skip4 = self.enc4(x3)   # skip4: 16x16x256, x4: 8x8x256
        
        # Bottleneck
        x = self.bottleneck(x4)     # 8x8x256
        
        # Decoder path with attention-gated skip connections
        x = self.dec4(x, skip4)     # 8x8 → 16x16, concat with skip4
        dec_out = self.dec3(x, skip3)     # 16x16 → 32x32, concat with skip3
        
        # Final convolution for image prediction
        mu = self.final_conv(dec_out)      # 32x32x3
        
        # Transform prediction from final decoder features
        transform_out = self.transform_head(dec_out)  # 4 scalars
        
        # Apply activation to image output
        if self.final_activation == 'sigmoid':
            mu = torch.sigmoid(mu)
        elif self.final_activation == 'tanh':
            mu = torch.tanh(mu)
        
        # Process transform predictions
        x_scale = torch.tanh(transform_out[:, 0])  # Range [-1, 1]
        y_scale = torch.tanh(transform_out[:, 1])  # Range [-1, 1]
        x_offset = torch.tanh(transform_out[:, 2])  # Range [-1, 1]
        y_offset = torch.tanh(transform_out[:, 3])  # Range [-1, 1]
        
        transform_params = torch.stack([x_scale, y_scale, x_offset, y_offset], dim=1)
        
        return mu, transform_params


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_loss(mu, target):
    """
    Simple MSE loss function for image reconstruction.
    
    Args:
        mu: Predicted values (B, C, H, W)
        target: Ground truth values (B, C, H, W)
    
    Returns:
        Loss value
    """
    return torch.nn.functional.mse_loss(mu, target)


def test_model():
    """Test the model architecture with random input."""
    model = MediumUNet()
    x = torch.randn(2, 3, 128, 128)  # Batch of 2, 3 channels, 128x128
    
    print(f"Input shape: {x.shape}")
    print(f"MediumUNet parameters: {count_parameters(model):,}")
    
    with torch.no_grad():
        mu, transform_params = model(x)
    
    print(f"Mu shape: {mu.shape}, Transform params shape: {transform_params.shape}")
    print(f"Mu range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"Transform params range: [{transform_params.min():.3f}, {transform_params.max():.3f}]")
    
    # Test loss function
    target = torch.randn_like(mu)
    loss = mse_loss(mu, target)
    print(f"MSE loss: {loss:.3f}")
    
    # Memory usage estimate
    def get_model_memory_usage(model, input_size):
        """Estimate memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # Rough estimate of activations (this is approximate)
        activation_size = 0
        with torch.no_grad():
            x = torch.randn(1, *input_size)
            # This is a rough estimate - actual memory usage will be higher during training
            _ = model(x)
        
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    memory_mb = get_model_memory_usage(model, (3, 128, 128))
    print(f"Estimated model memory usage: {memory_mb:.1f} MB")


if __name__ == "__main__":
    test_model()
