#!/usr/bin/env python3
"""
Medium U-Net architecture for patch-based denoising from 64x64 inputs.

Input: 64x64x3 RGB patches (noisy images)
Output: 16x16x3 RGB patches (center region prediction)

This is a more sophisticated version of CompactUNet with:
- Deeper network (4 encoder levels)
- More channels at each level
- Residual blocks with GroupNorm
- Attention gates in decoder
- Better transform detection head

Estimated parameters: ~2-3M (vs ~1.35M for CompactUNet)
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


class MediumUNet64x16(nn.Module):
    """
    Medium U-Net for center patch prediction from 64x64 patches.
    
    Architecture:
    - Encoder: 64x64 → 32x32 → 16x16 → 8x8
    - Bottleneck: 8x8 with 256 channels and residual blocks
    - Decoder: 8x8 → 16x16 (exact 16x16 output)
    
    Estimated parameters: ~2-3M
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid', use_attention: bool = True):
        super(MediumUNet64x16, self).__init__()
        
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
        
        # Apply activation to image output
        if self.final_activation == 'sigmoid':
            mu = torch.sigmoid(mu)
        elif self.final_activation == 'tanh':
            mu = torch.tanh(mu)
        elif self.final_activation == 'relu':
            mu = torch.relu(mu)
        
        return mu


class SpatialSelfAttention(nn.Module):
    """
    Spatial self-attention module for pixel art reconstruction.
    Helps maintain region coherence and sharp boundaries.
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super(SpatialSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)
        
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        # Output projection
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            out: Attention-enhanced feature map (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # Generate Q, K, V
        query = self.query_conv(x).view(B, self.inter_channels, -1)  # (B, C', H*W)
        key = self.key_conv(x).view(B, self.inter_channels, -1)      # (B, C', H*W)
        value = self.value_conv(x).view(B, self.inter_channels, -1)  # (B, C', H*W)
        
        # Compute attention weights
        query = query.permute(0, 2, 1)  # (B, H*W, C')
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = self.softmax(attention)
        
        # Apply attention to values
        value = value.permute(0, 2, 1)  # (B, H*W, C')
        out = torch.bmm(attention, value)  # (B, H*W, C')
        out = out.permute(0, 2, 1).view(B, self.inter_channels, H, W)  # (B, C', H, W)
        
        # Output projection
        out = self.out_conv(out)  # (B, C, H, W)
        
        # Residual connection with learnable scaling
        out = self.gamma * out + x
        
        return out


class ResidualBlockWithAttention(nn.Module):
    """Residual block with optional spatial self-attention."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, 
                 use_attention: bool = False, attention_reduction: int = 8):
        super(ResidualBlockWithAttention, self).__init__()
        
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
        
        # Optional spatial attention
        self.attention = SpatialSelfAttention(out_channels, attention_reduction) if use_attention else None
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm_act1(out)
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm_act2(out)
        
        # Add residual
        out = out + residual
        
        # Apply spatial attention if enabled
        if self.attention is not None:
            out = self.attention(out)
        
        return out


class MediumUNetSpatialAttention64x16(nn.Module):
    """
    Medium U-Net with spatial self-attention for center patch prediction from 64x64 patches.
    
    Architecture:
    - Encoder: 64x64 → 32x32 → 16x16 → 8x8
    - Bottleneck: 8x8 with 128 channels, residual blocks, and spatial attention
    - Decoder: 8x8 → 16x16 (exact 16x16 output) with spatial attention
    
    Key improvements over MediumUNet64x16:
    - Spatial self-attention in bottleneck and decoder for region coherence
    - Attention-enhanced residual blocks for better boundary preservation
    
    Estimated parameters: ~2-4M
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid', use_attention: bool = True,
                 attention_reduction: int = 8):
        super(MediumUNetSpatialAttention64x16, self).__init__()
        
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
        
        # Apply activation to image output
        if self.final_activation == 'sigmoid':
            mu = torch.sigmoid(mu)
        elif self.final_activation == 'tanh':
            mu = torch.tanh(mu)
        elif self.final_activation == 'relu':
            mu = torch.relu(mu)
        
        return mu


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
    """Test both model architectures with random input."""
    x = torch.randn(2, 3, 64, 64)  # Batch of 2, 3 channels, 64x64
    print(f"Input shape: {x.shape}")
    print("=" * 60)
    
    # Test original MediumUNet64x16
    print("Testing MediumUNet64x16:")
    model1 = MediumUNet64x16()
    print(f"Parameters: {count_parameters(model1):,}")
    
    with torch.no_grad():
        mu1 = model1(x)
    
    print(f"Output shape: {mu1.shape}")
    print(f"Output range: [{mu1.min():.3f}, {mu1.max():.3f}]")
    
    # Test loss function
    target = torch.randn_like(mu1)
    loss1 = mse_loss(mu1, target)
    print(f"MSE loss: {loss1:.3f}")
    
    print("-" * 40)
    
    # Test spatial attention version
    print("Testing MediumUNetSpatialAttention64x16:")
    model2 = MediumUNetSpatialAttention64x16()
    print(f"Parameters: {count_parameters(model2):,}")
    
    with torch.no_grad():
        mu2 = model2(x)
    
    print(f"Output shape: {mu2.shape}")
    print(f"Output range: [{mu2.min():.3f}, {mu2.max():.3f}]")
    
    loss2 = mse_loss(mu2, target)
    print(f"MSE loss: {loss2:.3f}")
    
    # Memory usage estimate
    def get_model_memory_usage(model, input_size):
        """Estimate memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    memory_mb1 = get_model_memory_usage(model1, (3, 64, 64))
    memory_mb2 = get_model_memory_usage(model2, (3, 64, 64))
    
    print("-" * 40)
    print("Memory Usage Comparison:")
    print(f"Original model: {memory_mb1:.1f} MB")
    print(f"Spatial attention model: {memory_mb2:.1f} MB")
    print(f"Additional memory for attention: {memory_mb2 - memory_mb1:.1f} MB")
    
    print("=" * 60)
    print("Summary:")
    print(f"Parameter increase: {count_parameters(model2) - count_parameters(model1):,} parameters")
    print(f"Relative increase: {(count_parameters(model2) / count_parameters(model1) - 1) * 100:.1f}%")


if __name__ == "__main__":
    test_model()


# Export the models for easy importing
__all__ = [
    'MediumUNet64x16', 
    'MediumUNetSpatialAttention64x16',
    'count_parameters',
    'mse_loss'
]
