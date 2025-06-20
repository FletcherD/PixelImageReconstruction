#!/usr/bin/env python3
"""
Modified U-Net architecture for single-pixel prediction from degraded pixel art patches.

Input: 128x128x3 RGB patches (degraded pixel art)
Output: 1x1x3 RGB values (single reconstructed pixel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


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
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class PixelUNet(nn.Module):
    """
    Modified U-Net for single pixel prediction.
    
    Architecture:
    - Encoder: 128x128 → 64x64 → 32x32 → 16x16 → 8x8
    - Decoder: 8x8 → 16x16 → 32x32 → 64x64 → 32x32 → 16x16 → 8x8 → 4x4 → 2x2 → 1x1
    - Skip connections at 64x64, 32x32, 16x16 resolutions
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid'):
        super(PixelUNet, self).__init__()
        
        self.final_activation = final_activation
        
        # Encoder (downsampling path)
        # 128x128 → 64x64
        self.enc1 = EncoderBlock(in_channels, 64)
        # 64x64 → 32x32  
        self.enc2 = EncoderBlock(64, 128)
        # 32x32 → 16x16
        self.enc3 = EncoderBlock(128, 256)
        # 16x16 → 8x8
        self.enc4 = EncoderBlock(256, 512, dropout=dropout)
        
        # Bottleneck at 8x8
        self.bottleneck = ConvBlock(512, 1024, dropout=dropout)
        
        # Decoder (upsampling path with progressive reduction)
        # 8x8 → 16x16 (with skip from enc4: 512 channels)
        self.dec4 = DecoderBlock(1024, 512, 256, dropout=dropout, use_transpose=use_transpose)
        # 16x16 → 32x32 (with skip from enc3: 256 channels)
        self.dec3 = DecoderBlock(256, 256, 128, dropout=dropout, use_transpose=use_transpose)
        # 32x32 → 64x64 (with skip from enc2: 128 channels)
        self.dec2 = DecoderBlock(128, 128, 64, use_transpose=use_transpose)
        
        # Additional reduction layers to get from 64x64 to 1x1
        # 64x64 → 32x32
        self.reduce1 = nn.Sequential(
            ConvBlock(64, 32),
            nn.MaxPool2d(2, 2)
        )
        # 32x32 → 16x16
        self.reduce2 = nn.Sequential(
            ConvBlock(32, 16),
            nn.MaxPool2d(2, 2)
        )
        # 16x16 → 8x8
        self.reduce3 = nn.Sequential(
            ConvBlock(16, 8),
            nn.MaxPool2d(2, 2)
        )
        # 8x8 → 4x4
        self.reduce4 = nn.Sequential(
            ConvBlock(8, 8),
            nn.MaxPool2d(2, 2)
        )
        # 4x4 → 2x2
        self.reduce5 = nn.Sequential(
            ConvBlock(8, 8),
            nn.MaxPool2d(2, 2)
        )
        # 2x2 → 1x1
        self.reduce6 = nn.Sequential(
            ConvBlock(8, 8),
            nn.MaxPool2d(2, 2)
        )
        
        # Final output layer
        self.final_conv = nn.Conv2d(8, out_channels, 1)
        
        # Alternative: Global Average Pooling approach
        self.use_gap = False  # Set to True to use GAP instead of progressive reduction
        # Always initialize GAP layers for potential use
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder path
        x1, skip1 = self.enc1(x)    # skip1: 128x128x64, x1: 64x64x64
        x2, skip2 = self.enc2(x1)   # skip2: 64x64x128, x2: 32x32x128
        x3, skip3 = self.enc3(x2)   # skip3: 32x32x256, x3: 16x16x256
        x4, skip4 = self.enc4(x3)   # skip4: 16x16x512, x4: 8x8x512
        
        # Bottleneck
        x = self.bottleneck(x4)     # 8x8x1024
        
        # Decoder path with skip connections
        x = self.dec4(x, skip4)     # 8x8 → 16x16, concat with skip4
        x = self.dec3(x, skip3)     # 16x16 → 32x32, concat with skip3
        x = self.dec2(x, skip2)     # 32x32 → 64x64, concat with skip2
        
        if self.use_gap:
            # Alternative approach: Global Average Pooling
            x = self.gap(x)         # 64x64 → 1x1
            x = self.gap_conv(x)    # Convert to output channels
        else:
            # Progressive reduction approach
            x = self.reduce1(x)     # 64x64 → 32x32
            x = self.reduce2(x)     # 32x32 → 16x16
            x = self.reduce3(x)     # 16x16 → 8x8
            x = self.reduce4(x)     # 8x8 → 4x4
            x = self.reduce5(x)     # 4x4 → 2x2
            x = self.reduce6(x)     # 2x2 → 1x1
            x = self.final_conv(x)  # Convert to output channels
        
        # Final activation
        if self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        # If 'none', return raw logits
        
        return x


class PixelUNetGAP(PixelUNet):
    """PixelUNet variant using Global Average Pooling for final reduction."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid'):
        # Initialize parent first
        super().__init__(in_channels, out_channels, dropout, use_transpose, final_activation)
        # Then set use_gap to True to enable GAP
        self.use_gap = True


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model architecture with random input."""
    # Test standard model
    model = PixelUNet()
    x = torch.randn(2, 3, 128, 128)  # Batch of 2, 3 channels, 128x128
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {count_parameters(model):,}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test GAP variant
    model_gap = PixelUNetGAP()
    print(f"\nGAP model parameters: {count_parameters(model_gap):,}")
    
    with torch.no_grad():
        output_gap = model_gap(x)
    
    print(f"GAP output shape: {output_gap.shape}")
    print(f"GAP output range: [{output_gap.min():.3f}, {output_gap.max():.3f}]")


if __name__ == "__main__":
    test_model()