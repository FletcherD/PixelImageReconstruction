#!/usr/bin/env python3
"""
Exact color U-Net models that work with RGB values in [0,255] range.

These models avoid all color normalization and work directly with exact RGB integer values,
preventing the subtle color shifts that occur with standard [0,1] normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import the existing architectural components
from medium_unet import (
    GroupNormAct, ResidualBlock, SEBlock, AttentionGate, EncoderBlock, DecoderBlock,
    SpatialSelfAttention, ResidualBlockWithAttention
)


class ExactColorMediumUNet64x16(nn.Module):
    """
    Medium U-Net for exact color preservation with 64x64 input → 16x16 output.
    
    Key differences from standard MediumUNet64x16:
    - Works with RGB values in [0, 255] range
    - Uses scaled sigmoid activation (0-255) or no activation
    - Designed for exact color matching in pixel art
    
    Architecture:
    - Encoder: 64x64 → 32x32 → 16x16 → 8x8
    - Bottleneck: 8x8 with 128 channels and residual blocks
    - Decoder: 8x8 → 16x16 (exact 16x16 output)
    
    Estimated parameters: ~1.6M
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'scaled_sigmoid', use_attention: bool = True):
        super(ExactColorMediumUNet64x16, self).__init__()
        
        self.final_activation = final_activation
        
        # Encoder path - 3 levels for 64x64 input
        # 64x64 → 32x32
        self.enc1 = EncoderBlock(in_channels, 32, dropout * 0.5)
        # 32x32 → 16x16  
        self.enc2 = EncoderBlock(32, 64, dropout * 0.7)
        # 16x16 → 8x8
        self.enc3 = EncoderBlock(64, 128, dropout * 0.8)
        
        # Bottleneck at 8x8 with 128 channels
        self.bottleneck = nn.Sequential(
            ResidualBlock(128, 128, dropout),
            ResidualBlock(128, 128, dropout),
            SEBlock(128)
        )
        
        # Decoder path - to exact 16x16 output
        # 8x8 → 16x16 (with skip from enc3: 128 channels)
        self.dec3 = DecoderBlock(128, 128, 64, dropout, use_attention, use_transpose)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            GroupNormAct(32),
            nn.Conv2d(32, out_channels, 1)
        )
    
    def forward(self, x):
        # Encoder path
        x1, skip1 = self.enc1(x)    # skip1: 64x64x32, x1: 32x32x32
        x2, skip2 = self.enc2(x1)   # skip2: 32x32x64, x2: 16x16x64
        x3, skip3 = self.enc3(x2)   # skip3: 16x16x128, x3: 8x8x128
        
        # Bottleneck
        x = self.bottleneck(x3)     # 8x8x128
        
        # Decoder path with attention-gated skip connections
        dec_out = self.dec3(x, skip3)     # 8x8 → 16x16, concat with skip3
        
        # Final convolution for image prediction
        mu = self.final_conv(dec_out)      # 16x16x3
        
        # Apply activation to image output for [0, 255] range
        if self.final_activation == 'scaled_sigmoid':
            mu = torch.sigmoid(mu) * 255.0  # Scale sigmoid to [0, 255]
        elif self.final_activation == 'relu':
            mu = torch.clamp(torch.relu(mu), 0, 255)  # Clamp ReLU to [0, 255]
        elif self.final_activation == 'tanh':
            mu = (torch.tanh(mu) + 1.0) * 127.5  # Scale tanh from [-1,1] to [0,255]
        elif self.final_activation == 'clamp':
            mu = torch.clamp(mu, 0, 255)  # Just clamp raw output
        # elif self.final_activation == 'none': no activation applied
        
        return mu


class ExactColorMediumUNetSpatialAttention64x16(nn.Module):
    """
    Medium U-Net with spatial attention for exact color preservation.
    
    Combines spatial self-attention with exact RGB value processing
    for superior pixel art reconstruction quality.
    
    Key features:
    - Spatial self-attention for region coherence
    - Works with RGB values in [0, 255] range
    - Preserves exact colors throughout processing
    
    Estimated parameters: ~1.6-1.8M
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'scaled_sigmoid', use_attention: bool = True,
                 attention_reduction: int = 8):
        super(ExactColorMediumUNetSpatialAttention64x16, self).__init__()
        
        self.final_activation = final_activation
        
        # Encoder path - 3 levels for 64x64 input (reuse existing EncoderBlock)
        # 64x64 → 32x32
        self.enc1 = EncoderBlock(in_channels, 32, dropout * 0.5)
        # 32x32 → 16x16  
        self.enc2 = EncoderBlock(32, 64, dropout * 0.7)
        # 16x16 → 8x8
        self.enc3 = EncoderBlock(64, 128, dropout * 0.8)
        
        # Enhanced bottleneck at 8x8 with spatial attention
        self.bottleneck = nn.Sequential(
            ResidualBlockWithAttention(128, 128, dropout, use_attention=True, 
                                     attention_reduction=attention_reduction),
            ResidualBlockWithAttention(128, 128, dropout, use_attention=True, 
                                     attention_reduction=attention_reduction),
            SEBlock(128)
        )
        
        # Decoder path with spatial attention - to exact 16x16 output
        # 8x8 → 16x16 (with skip from enc3: 128 channels)
        self.dec3 = DecoderBlock(128, 128, 64, dropout, use_attention, use_transpose)
        
        # Spatial attention in decoder features before final conv
        self.decoder_attention = SpatialSelfAttention(64, attention_reduction) if use_attention else None
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            GroupNormAct(32),
            nn.Conv2d(32, out_channels, 1)
        )
    
    def forward(self, x):
        # Encoder path (reuse existing encoder blocks)
        x1, skip1 = self.enc1(x)    # skip1: 64x64x32, x1: 32x32x32
        x2, skip2 = self.enc2(x1)   # skip2: 32x32x64, x2: 16x16x64
        x3, skip3 = self.enc3(x2)   # skip3: 16x16x128, x3: 8x8x128
        
        # Enhanced bottleneck with spatial attention
        x = self.bottleneck(x3)     # 8x8x128
        
        # Decoder path with attention-gated skip connections
        dec_out = self.dec3(x, skip3)     # 8x8 → 16x16, concat with skip3
        
        # Apply spatial attention to decoder features
        if self.decoder_attention is not None:
            dec_out = self.decoder_attention(dec_out)
        
        # Final convolution for image prediction
        mu = self.final_conv(dec_out)      # 16x16x3
        
        # Apply activation to image output for [0, 255] range
        if self.final_activation == 'scaled_sigmoid':
            mu = torch.sigmoid(mu) * 255.0  # Scale sigmoid to [0, 255]
        elif self.final_activation == 'relu':
            mu = torch.clamp(torch.relu(mu), 0, 255)  # Clamp ReLU to [0, 255]
        elif self.final_activation == 'tanh':
            mu = (torch.tanh(mu) + 1.0) * 127.5  # Scale tanh from [-1,1] to [0,255]
        elif self.final_activation == 'clamp':
            mu = torch.clamp(mu, 0, 255)  # Just clamp raw output
        # elif self.final_activation == 'none': no activation applied
        
        return mu


def exact_color_mse_loss(pred, target):
    """
    MSE loss for exact color models working in [0, 255] range.
    
    Args:
        pred: Predicted RGB values in [0, 255] range
        target: Target RGB values in [0, 255] range
    
    Returns:
        MSE loss value
    """
    return F.mse_loss(pred, target)


def exact_color_l1_loss(pred, target):
    """
    L1 (MAE) loss for exact color models working in [0, 255] range.
    
    Args:
        pred: Predicted RGB values in [0, 255] range
        target: Target RGB values in [0, 255] range
    
    Returns:
        L1 loss value
    """
    return F.l1_loss(pred, target)


def exact_color_huber_loss(pred, target, delta: float = 1.0):
    """
    Huber loss for exact color models working in [0, 255] range.
    
    Huber loss is less sensitive to outliers than MSE.
    
    Args:
        pred: Predicted RGB values in [0, 255] range
        target: Target RGB values in [0, 255] range
        delta: Threshold for switching between L1 and L2 loss
    
    Returns:
        Huber loss value
    """
    return F.huber_loss(pred, target, delta=delta)


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_exact_color_models():
    """Test both exact color model architectures."""
    
    print("Testing Exact Color Models")
    print("=" * 50)
    
    # Test input with RGB values in [0, 255] range
    x = torch.randint(0, 256, (2, 3, 64, 64)).float()  # Random RGB values
    target = torch.randint(0, 256, (2, 3, 16, 16)).float()  # Random target
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.0f}, {x.max():.0f}]")
    print(f"Target shape: {target.shape}")
    print(f"Target range: [{target.min():.0f}, {target.max():.0f}]")
    print()
    
    # Test basic exact color model
    print("Testing ExactColorMediumUNet64x16:")
    model1 = ExactColorMediumUNet64x16(final_activation='scaled_sigmoid')
    print(f"Parameters: {count_parameters(model1):,}")
    
    with torch.no_grad():
        pred1 = model1(x)
    
    print(f"Output shape: {pred1.shape}")
    print(f"Output range: [{pred1.min():.1f}, {pred1.max():.1f}]")
    
    # Test loss
    loss1 = exact_color_mse_loss(pred1, target)
    print(f"MSE loss: {loss1:.1f}")
    
    print("-" * 30)
    
    # Test spatial attention exact color model
    print("Testing ExactColorMediumUNetSpatialAttention64x16:")
    model2 = ExactColorMediumUNetSpatialAttention64x16(final_activation='scaled_sigmoid')
    print(f"Parameters: {count_parameters(model2):,}")
    
    with torch.no_grad():
        pred2 = model2(x)
    
    print(f"Output shape: {pred2.shape}")
    print(f"Output range: [{pred2.min():.1f}, {pred2.max():.1f}]")
    
    # Test loss
    loss2 = exact_color_mse_loss(pred2, target)
    print(f"MSE loss: {loss2:.1f}")
    
    print("-" * 30)
    
    # Test different activation functions
    print("Testing different activation functions:")
    activations = ['scaled_sigmoid', 'relu', 'tanh', 'clamp', 'none']
    
    for activation in activations:
        model_test = ExactColorMediumUNet64x16(final_activation=activation)
        with torch.no_grad():
            pred_test = model_test(x)
        print(f"  {activation:15}: [{pred_test.min():6.1f}, {pred_test.max():6.1f}]")
    
    print()
    print("✓ All exact color models tested successfully!")
    
    print("\nModel Comparison:")
    print(f"Basic model parameters:     {count_parameters(model1):,}")
    print(f"Spatial attention model:    {count_parameters(model2):,}")
    print(f"Parameter increase:         {count_parameters(model2) - count_parameters(model1):,}")
    print(f"Relative increase:          {(count_parameters(model2) / count_parameters(model1) - 1) * 100:.1f}%")


if __name__ == "__main__":
    test_exact_color_models()