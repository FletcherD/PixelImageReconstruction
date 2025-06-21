#!/usr/bin/env python3
"""
Compact U-Net architecture for patch-based denoising from 128x128 inputs.

Input: 128x128x3 RGB patches (noisy images)
Output: 32x32x3 RGB patches (center region prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Basic convolutional block with Conv2d -> BatchNorm -> ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 padding: int = 1, dropout: float = 0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block with conv block followed by max pooling."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and conv block."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, 
                 dropout: float = 0.0, use_transpose: bool = True):
        super(DecoderBlock, self).__init__()
        self.use_transpose = use_transpose
        
        if use_transpose:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.upsample_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        
        # Concatenated channels: upsampled + skip connection
        concat_channels = (in_channels // 2) + skip_channels
        self.conv_block = ConvBlock(concat_channels, out_channels, dropout=dropout)
    
    def forward(self, x, skip):
        if self.use_transpose:
            x = self.upsample(x)
        else:
            x = self.upsample(x)
            x = self.upsample_conv(x)
        
        # Handle size mismatch due to odd input dimensions
        if x.size() != skip.size():
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class CompactUNet(nn.Module):
    """
    Compact U-Net for center patch prediction from 128x128 patches.
    
    Architecture:
    - Encoder: 128x128 → 64x64 → 32x32 → 16x16 → 8x8
    - Bottleneck: 8x8 → 8x8 (256 channels)
    - Decoder: 8x8 → 16x16 → 32x32 (exact 32x32 output)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid'):
        super(CompactUNet, self).__init__()
        
        self.final_activation = final_activation
        
        # Encoder (downsampling path) - compact channel progression
        # 128x128 → 64x64
        self.enc1 = EncoderBlock(in_channels, 16)
        # 64x64 → 32x32  
        self.enc2 = EncoderBlock(16, 32)
        # 32x32 → 16x16
        self.enc3 = EncoderBlock(32, 64)
        # 16x16 → 8x8
        self.enc4 = EncoderBlock(64, 128, dropout=dropout)
        
        # Bottleneck at 8x8
        self.bottleneck = nn.Sequential(
            ConvBlock(128, 256, dropout=dropout),
            ConvBlock(256, 256, dropout=dropout)
        )
        
        # Decoder (upsampling path) - to exact 32x32 output
        # 8x8 → 16x16 (with skip from enc4: 128 channels)
        self.dec4 = DecoderBlock(256, 128, 128, dropout=dropout, use_transpose=use_transpose)
        # 16x16 → 32x32 (with skip from enc3: 64 channels)
        self.dec3 = DecoderBlock(128, 64, 64, dropout=dropout, use_transpose=use_transpose)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder path
        x1, skip1 = self.enc1(x)    # skip1: 128x128x16, x1: 64x64x16
        x2, skip2 = self.enc2(x1)   # skip2: 64x64x32, x2: 32x32x32
        x3, skip3 = self.enc3(x2)   # skip3: 32x32x64, x3: 16x16x64
        x4, skip4 = self.enc4(x3)   # skip4: 16x16x128, x4: 8x8x128
        
        # Bottleneck
        x = self.bottleneck(x4)     # 8x8x256
        
        # Decoder path with skip connections to exact 32x32
        x = self.dec4(x, skip4)     # 8x8 → 16x16, concat with skip4
        x = self.dec3(x, skip3)     # 16x16 → 32x32, concat with skip3
        
        # Final convolution
        x = self.final_conv(x)      # 32x32x3
        
        # Final activation
        if self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        # If 'none', return raw logits
        
        return x


class CompactUNetGAP(nn.Module):
    """
    Compact U-Net variant using Global Average Pooling for center patch prediction.
    More parameter-efficient alternative that uses encoder + GAP + upsampling instead of full U-Net.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, final_activation: str = 'sigmoid'):
        super(CompactUNetGAP, self).__init__()
        
        self.final_activation = final_activation
        
        # Encoder path - same as CompactUNet
        self.enc1 = EncoderBlock(in_channels, 16)
        self.enc2 = EncoderBlock(16, 32)
        self.enc3 = EncoderBlock(32, 64)
        self.enc4 = EncoderBlock(64, 128, dropout=dropout)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(128, 256, dropout=dropout),
            ConvBlock(256, 256, dropout=dropout)
        )
        
        # Global Average Pooling + MLP for feature extraction
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32*32*out_channels, 1)  # Predict flattened 32x32 patch
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder path
        x1, _ = self.enc1(x)    # 66x66x16
        x2, _ = self.enc2(x1)   # 33x33x32
        x3, _ = self.enc3(x2)   # 17x17x64
        x4, _ = self.enc4(x3)   # 9x9x128
        
        # Bottleneck
        x = self.bottleneck(x4) # 9x9x256
        
        # Global Average Pooling + MLP
        x = self.gap(x)         # 1x1x256
        x = self.mlp(x)         # 1x1x(32*32*3)
        
        # Reshape to 32x32 patch
        x = x.view(batch_size, 3, 32, 32)  # Shape: (batch_size, 3, 32, 32)
        
        # Final activation
        if self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        
        return x


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model architecture with random input."""
    # Test CompactUNet
    model = CompactUNet()
    x = torch.randn(2, 3, 128, 128)  # Batch of 2, 3 channels, 128x128
    
    print(f"Input shape: {x.shape}")
    print(f"CompactUNet parameters: {count_parameters(model):,}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test CompactUNetGAP
    model_gap = CompactUNetGAP()
    print(f"\nCompactUNetGAP parameters: {count_parameters(model_gap):,}")
    
    with torch.no_grad():
        output_gap = model_gap(x)
    
    print(f"GAP output shape: {output_gap.shape}")
    print(f"GAP output range: [{output_gap.min():.3f}, {output_gap.max():.3f}]")


if __name__ == "__main__":
    test_model()