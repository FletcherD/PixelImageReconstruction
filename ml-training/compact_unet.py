#!/usr/bin/env python3
"""
Compact U-Net architecture for patch-based denoising from 129x129 inputs.

Input: 129x129x3 RGB patches (noisy images)
Output: 1x1x3 RGB values (center pixel prediction)
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
    Compact U-Net for center pixel prediction from 129x129 patches.
    
    Architecture:
    - Encoder: 129x129 → 65x65 → 33x33 → 17x17 → 9x9
    - Bottleneck: 9x9 → 9x9 (256 channels)
    - Decoder: 9x9 → 17x17 → 33x33 → 65x65 → 129x129
    - Center pixel extraction
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid'):
        super(CompactUNet, self).__init__()
        
        self.final_activation = final_activation
        
        # Encoder (downsampling path) - compact channel progression
        # 129x129 → 65x65
        self.enc1 = EncoderBlock(in_channels, 16)
        # 65x65 → 33x33  
        self.enc2 = EncoderBlock(16, 32)
        # 33x33 → 17x17
        self.enc3 = EncoderBlock(32, 64)
        # 17x17 → 9x9
        self.enc4 = EncoderBlock(64, 128, dropout=dropout)
        
        # Bottleneck at 9x9
        self.bottleneck = nn.Sequential(
            ConvBlock(128, 256, dropout=dropout),
            ConvBlock(256, 256, dropout=dropout)
        )
        
        # Decoder (upsampling path)
        # 9x9 → 17x17 (with skip from enc4: 128 channels)
        self.dec4 = DecoderBlock(256, 128, 128, dropout=dropout, use_transpose=use_transpose)
        # 17x17 → 33x33 (with skip from enc3: 64 channels)
        self.dec3 = DecoderBlock(128, 64, 64, dropout=dropout, use_transpose=use_transpose)
        # 33x33 → 65x65 (with skip from enc2: 32 channels)
        self.dec2 = DecoderBlock(64, 32, 32, use_transpose=use_transpose)
        # 65x65 → 129x129 (with skip from enc1: 16 channels)
        self.dec1 = DecoderBlock(32, 16, 16, use_transpose=use_transpose)
        
        # Final output layer
        self.final_conv = nn.Conv2d(16, out_channels, 1)
    
    def forward(self, x):
        # Store input size for center pixel extraction
        batch_size = x.size(0)
        center_idx = x.size(2) // 2  # Should be 64 for 129x129 input
        
        # Encoder path
        x1, skip1 = self.enc1(x)    # skip1: 129x129x16, x1: 65x65x16
        x2, skip2 = self.enc2(x1)   # skip2: 65x65x32, x2: 33x33x32
        x3, skip3 = self.enc3(x2)   # skip3: 33x33x64, x3: 17x17x64
        x4, skip4 = self.enc4(x3)   # skip4: 17x17x128, x4: 9x9x128
        
        # Bottleneck
        x = self.bottleneck(x4)     # 9x9x256
        
        # Decoder path with skip connections
        x = self.dec4(x, skip4)     # 9x9 → 17x17, concat with skip4
        x = self.dec3(x, skip3)     # 17x17 → 33x33, concat with skip3
        x = self.dec2(x, skip2)     # 33x33 → 65x65, concat with skip2
        x = self.dec1(x, skip1)     # 65x65 → 129x129, concat with skip1
        
        # Final convolution
        x = self.final_conv(x)      # 129x129x3
        
        # Extract center pixel
        center_pixel = x[:, :, center_idx, center_idx]  # Shape: (batch_size, channels)
        center_pixel = center_pixel.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, channels, 1, 1)
        
        # Final activation
        if self.final_activation == 'sigmoid':
            center_pixel = torch.sigmoid(center_pixel)
        elif self.final_activation == 'tanh':
            center_pixel = torch.tanh(center_pixel)
        # If 'none', return raw logits
        
        return center_pixel


class CompactUNetGAP(nn.Module):
    """
    Compact U-Net variant using Global Average Pooling for center pixel prediction.
    More parameter-efficient alternative that uses encoder + GAP instead of full U-Net.
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
        
        # Global Average Pooling + MLP for center pixel prediction
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )
    
    def forward(self, x):
        # Encoder path
        x1, _ = self.enc1(x)    # 65x65x16
        x2, _ = self.enc2(x1)   # 33x33x32
        x3, _ = self.enc3(x2)   # 17x17x64
        x4, _ = self.enc4(x3)   # 9x9x128
        
        # Bottleneck
        x = self.bottleneck(x4) # 9x9x256
        
        # Global Average Pooling + MLP
        x = self.gap(x)         # 1x1x256
        x = self.mlp(x)         # 1x1x3
        
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
    x = torch.randn(2, 3, 129, 129)  # Batch of 2, 3 channels, 129x129
    
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