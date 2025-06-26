#!/usr/bin/env python3
"""
Palette-based U-Net architecture for palette classification from 128x128 inputs.

Input: 128x128x3 RGB patches (noisy images)
Output: 32x32x64 logits (palette classification for each pixel)

This model classifies each pixel into one of 64 predefined palette colors
instead of predicting RGB values directly. Based on MediumUNet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

# Import components from MediumUNet
from medium_unet import (
    GroupNormAct, ResidualBlock, SEBlock, AttentionGate, 
    EncoderBlock, DecoderBlock
)


class PaletteUNet(nn.Module):
    """
    Palette-based U-Net for pixel classification into 64 palette colors.
    
    Architecture:
    - Encoder: 128x128 → 64x64 → 32x32 → 16x16 → 8x8 → 4x4
    - Bottleneck: 4x4 with 512 channels and residual blocks
    - Decoder: 4x4 → 8x8 → 16x16 → 32x32 (exact 32x32 output)
    - Output: 32x32x64 logits for palette classification
    
    Estimated parameters: ~9-10M (similar to MediumUNet)
    """
    
    def __init__(self, in_channels: int = 3, num_palette_colors: int = 64, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 use_attention: bool = True):
        super(PaletteUNet, self).__init__()
        
        self.num_palette_colors = num_palette_colors
        
        # Encoder path - 4 levels (same as MediumUNet)
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
        
        # Final classification layer - outputs palette logits
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            GroupNormAct(32),
            nn.Conv2d(32, num_palette_colors, 1)  # 64 channels for 64 palette colors
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
        
        # Final convolution for palette classification
        logits = self.final_conv(dec_out)      # 32x32x64
        
        return logits


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def palette_classification_loss(logits, target_indices):
    """
    Cross-entropy loss for palette classification.
    
    Args:
        logits: Predicted logits (B, num_palette_colors, H, W)
        target_indices: Ground truth palette indices (B, H, W)
    
    Returns:
        Loss value
    """
    return F.cross_entropy(logits, target_indices)


def palette_accuracy(logits, target_indices):
    """
    Calculate classification accuracy for palette prediction.
    
    Args:
        logits: Predicted logits (B, num_palette_colors, H, W)
        target_indices: Ground truth palette indices (B, H, W)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    with torch.no_grad():
        predicted_indices = torch.argmax(logits, dim=1)
        correct = (predicted_indices == target_indices).float()
        return correct.mean().item()


def logits_to_palette_indices(logits):
    """
    Convert logits to palette indices using argmax.
    
    Args:
        logits: Predicted logits (B, num_palette_colors, H, W)
    
    Returns:
        Palette indices (B, H, W)
    """
    return torch.argmax(logits, dim=1)


def palette_indices_to_rgb(indices, palette_tensor):
    """
    Convert palette indices to RGB images using a palette.
    
    Args:
        indices: Palette indices (B, H, W)
        palette_tensor: Palette colors (num_colors, 3) in range [0, 1]
    
    Returns:
        RGB images (B, 3, H, W) in range [0, 1]
    """
    b, h, w = indices.shape
    device = indices.device
    
    # Ensure palette is on the same device
    if palette_tensor.device != device:
        palette_tensor = palette_tensor.to(device)
    
    # Convert indices to RGB
    rgb_flat = palette_tensor[indices.view(-1)]  # (B*H*W, 3)
    rgb = rgb_flat.view(b, h, w, 3)  # (B, H, W, 3)
    
    # Convert to (B, 3, H, W) format
    rgb = rgb.permute(0, 3, 1, 2)
    
    return rgb


def test_model():
    """Test the model architecture with random input."""
    model = PaletteUNet()
    x = torch.randn(2, 3, 128, 128)  # Batch of 2, 3 channels, 128x128
    
    print(f"Input shape: {x.shape}")
    print(f"PaletteUNet parameters: {count_parameters(model):,}")
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test loss function with random targets
    target_indices = torch.randint(0, 64, (2, 32, 32))
    loss = palette_classification_loss(logits, target_indices)
    print(f"Classification loss: {loss:.3f}")
    
    # Test accuracy
    accuracy = palette_accuracy(logits, target_indices)
    print(f"Random accuracy: {accuracy:.3f}")
    
    # Test index conversion
    predicted_indices = logits_to_palette_indices(logits)
    print(f"Predicted indices shape: {predicted_indices.shape}")
    print(f"Predicted indices range: [{predicted_indices.min()}, {predicted_indices.max()}]")
    
    # Test RGB conversion with a simple palette
    palette = torch.rand(64, 3)  # Random palette for testing
    rgb_output = palette_indices_to_rgb(predicted_indices, palette)
    print(f"RGB output shape: {rgb_output.shape}")
    print(f"RGB output range: [{rgb_output.min():.3f}, {rgb_output.max():.3f}]")
    
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