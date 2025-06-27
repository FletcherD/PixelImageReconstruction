#!/usr/bin/env python3
"""
Improved spatial attention mechanisms for pixel art reconstruction.

Addresses issues in the original implementation:
1. Better initialization of attention scaling
2. Temperature control for attention sharpness
3. Multi-scale attention application
4. Configurable attention strength
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Import existing components
from medium_unet import GroupNormAct, ResidualBlock, SEBlock, EncoderBlock, DecoderBlock


class ImprovedSpatialSelfAttention(nn.Module):
    """
    Improved spatial self-attention with better initialization and temperature control.
    """
    
    def __init__(self, in_channels: int, reduction: int = 4, 
                 temperature: float = 1.0, init_gamma: float = 0.1):
        super(ImprovedSpatialSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.temperature = temperature
        self.inter_channels = max(in_channels // reduction, 8)  # Minimum 8 channels
        
        # Query, Key, Value projections with proper initialization
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        # Output projection with residual scaling
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        # Learnable scaling parameter with better initialization
        self.gamma = nn.Parameter(torch.ones(1) * init_gamma)
        
        # Layer norm for better gradient flow
        self.layer_norm = nn.GroupNorm(8, in_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training dynamics."""
        # Initialize projection layers with Xavier uniform
        for module in [self.query_conv, self.key_conv, self.value_conv, self.out_conv]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            out: Attention-enhanced feature map (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # Apply layer norm to input
        x_norm = self.layer_norm(x)
        
        # Generate Q, K, V from normalized input
        query = self.query_conv(x_norm).view(B, self.inter_channels, -1)  # (B, C', H*W)
        key = self.key_conv(x_norm).view(B, self.inter_channels, -1)      # (B, C', H*W)
        value = self.value_conv(x_norm).view(B, self.inter_channels, -1)  # (B, C', H*W)
        
        # Compute attention weights with temperature scaling
        query = query.permute(0, 2, 1)  # (B, H*W, C')
        key = key.permute(0, 2, 1)      # (B, H*W, C')
        
        # Scaled dot-product attention with temperature
        attention_logits = torch.bmm(query, key.permute(0, 2, 1))  # (B, H*W, H*W)
        attention_logits = attention_logits / (self.temperature * math.sqrt(self.inter_channels))
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Apply attention to values
        value = value.permute(0, 2, 1)  # (B, H*W, C')
        attended = torch.bmm(attention_weights, value)  # (B, H*W, C')
        attended = attended.permute(0, 2, 1).view(B, self.inter_channels, H, W)  # (B, C', H, W)
        
        # Output projection
        attended = self.out_conv(attended)  # (B, C, H, W)
        
        # Residual connection with learnable scaling
        out = self.gamma * attended + x
        
        return out


class PositionalSpatialAttention(nn.Module):
    """
    Spatial attention with positional encoding for better spatial awareness.
    """
    
    def __init__(self, in_channels: int, reduction: int = 4, 
                 max_size: int = 64, temperature: float = 1.0):
        super(PositionalSpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.temperature = temperature
        self.inter_channels = max(in_channels // reduction, 8)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.inter_channels, max_size, max_size) * 0.02)
        
        # Q, K, V projections
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        # Output projection
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        # Attention strength parameter
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.query_conv, self.key_conv, self.value_conv, self.out_conv]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Add positional encoding
        pos_enc = self.pos_embedding[:, :, :H, :W]
        
        # Generate Q, K, V
        query = self.query_conv(x) + pos_enc  # Add position to query
        key = self.key_conv(x) + pos_enc      # Add position to key
        value = self.value_conv(x)
        
        # Reshape for attention computation
        query = query.view(B, self.inter_channels, -1).permute(0, 2, 1)  # (B, H*W, C')
        key = key.view(B, self.inter_channels, -1).permute(0, 2, 1)      # (B, H*W, C')
        value = value.view(B, self.inter_channels, -1).permute(0, 2, 1)  # (B, H*W, C')
        
        # Compute attention
        attention_logits = torch.bmm(query, key.permute(0, 2, 1)) / (self.temperature * math.sqrt(self.inter_channels))
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Apply attention
        attended = torch.bmm(attention_weights, value)  # (B, H*W, C')
        attended = attended.permute(0, 2, 1).view(B, self.inter_channels, H, W)
        
        # Output projection and residual
        out = self.out_conv(attended)
        out = self.gamma * out + x
        
        return out


class MultiScaleSpatialAttention(nn.Module):
    """
    Multi-scale spatial attention that operates at different resolutions.
    """
    
    def __init__(self, in_channels: int, scales: list = [1, 2, 4], 
                 reduction: int = 4, temperature: float = 1.0):
        super(MultiScaleSpatialAttention, self).__init__()
        self.scales = scales
        self.inter_channels = max(in_channels // reduction, 8)
        
        # Separate attention modules for each scale
        self.attention_modules = nn.ModuleList([
            ImprovedSpatialSelfAttention(in_channels, reduction, temperature, init_gamma=0.1/len(scales))
            for _ in scales
        ])
        
        # Scale-specific pooling
        self.pooling = nn.ModuleList([
            nn.AvgPool2d(scale, scale) if scale > 1 else nn.Identity()
            for scale in scales
        ])
        
        # Upsampling for multi-scale fusion
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False) if scale > 1 else nn.Identity()
            for scale in scales
        ])
        
        # Fusion weights
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(self, x):
        B, C, H, W = x.size()
        scale_outputs = []
        
        for i, (pool, attention, upsample) in enumerate(zip(self.pooling, self.attention_modules, self.upsample)):
            # Downsample if needed
            x_scaled = pool(x)
            
            # Apply attention at this scale
            attended = attention(x_scaled)
            
            # Upsample back to original resolution
            attended_upsampled = upsample(attended)
            
            # Handle size mismatch due to pooling/upsampling
            if attended_upsampled.size() != x.size():
                attended_upsampled = F.interpolate(attended_upsampled, size=(H, W), mode='bilinear', align_corners=False)
            
            scale_outputs.append(attended_upsampled)
        
        # Weighted fusion of multi-scale outputs
        weights = F.softmax(self.scale_weights, dim=0)
        fused_output = sum(w * out for w, out in zip(weights, scale_outputs))
        
        return fused_output


class AttentionAnalyzer:
    """
    Utility class to analyze attention patterns and effectiveness.
    """
    
    @staticmethod
    def compute_attention_entropy(attention_weights):
        """Compute entropy of attention weights to measure focus."""
        # attention_weights: (B, H*W, H*W)
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
        return entropy.mean()
    
    @staticmethod
    def compute_attention_locality(attention_weights, spatial_size):
        """Measure how local vs global the attention is."""
        B, HW, _ = attention_weights.shape
        H = W = int(math.sqrt(HW))
        
        # Create distance matrix
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)
        coords = coords.view(-1, 2).float()  # (H*W, 2)
        
        dist_matrix = torch.cdist(coords, coords)  # (H*W, H*W)
        
        # Weighted average distance
        avg_distance = (attention_weights * dist_matrix.unsqueeze(0)).sum(dim=-1).mean()
        return avg_distance
    
    @staticmethod
    def visualize_attention_patterns(attention_weights, input_size, save_path=None):
        """Visualize attention patterns for debugging."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        B, HW, _ = attention_weights.shape
        H = W = int(math.sqrt(HW))
        
        # Take first batch and a few spatial positions
        attn = attention_weights[0].view(H, W, H, W).cpu().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Show attention maps for different query positions
        query_positions = [(H//4, W//4), (H//2, W//2), (3*H//4, 3*W//4)]
        
        for i, (qh, qw) in enumerate(query_positions):
            attention_map = attn[qh, qw]
            axes[i].imshow(attention_map, cmap='hot', interpolation='nearest')
            axes[i].set_title(f'Attention from ({qh}, {qw})')
            axes[i].axis('off')
        
        # Show average attention pattern
        avg_attention = attn.mean(axis=(0, 1))
        axes[3].imshow(avg_attention, cmap='hot', interpolation='nearest')
        axes[3].set_title('Average Attention Pattern')
        axes[3].axis('off')
        
        # Show attention entropy map
        entropy_map = -(attn * np.log(attn + 1e-8)).sum(axis=(2, 3))
        axes[4].imshow(entropy_map, cmap='viridis', interpolation='nearest')
        axes[4].set_title('Attention Entropy (Focus)')
        axes[4].axis('off')
        
        # Show input image for reference
        axes[5].text(0.5, 0.5, 'Input Image\n(Reference)', ha='center', va='center', fontsize=12)
        axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def test_attention_improvements():
    """Test different attention mechanisms."""
    
    print("Testing improved spatial attention mechanisms...")
    
    # Test input
    x = torch.randn(2, 64, 16, 16)  # Typical decoder feature size
    
    print(f"Input shape: {x.shape}")
    
    # Test original attention
    from medium_unet import SpatialSelfAttention
    original_attn = SpatialSelfAttention(64, reduction=8)
    original_params = sum(p.numel() for p in original_attn.parameters())
    
    with torch.no_grad():
        original_out = original_attn(x)
    
    print(f"\nOriginal Attention:")
    print(f"  Parameters: {original_params:,}")
    print(f"  Gamma value: {original_attn.gamma.item():.6f}")
    print(f"  Output change: {torch.norm(original_out - x).item():.6f}")
    
    # Test improved attention
    improved_attn = ImprovedSpatialSelfAttention(64, reduction=4, temperature=0.5, init_gamma=0.2)
    improved_params = sum(p.numel() for p in improved_attn.parameters())
    
    with torch.no_grad():
        improved_out = improved_attn(x)
    
    print(f"\nImproved Attention:")
    print(f"  Parameters: {improved_params:,}")
    print(f"  Gamma value: {improved_attn.gamma.item():.6f}")
    print(f"  Output change: {torch.norm(improved_out - x).item():.6f}")
    
    # Test positional attention
    pos_attn = PositionalSpatialAttention(64, reduction=4, max_size=32)
    pos_params = sum(p.numel() for p in pos_attn.parameters())
    
    with torch.no_grad():
        pos_out = pos_attn(x)
    
    print(f"\nPositional Attention:")
    print(f"  Parameters: {pos_params:,}")
    print(f"  Gamma value: {pos_attn.gamma.item():.6f}")
    print(f"  Output change: {torch.norm(pos_out - x).item():.6f}")
    
    # Test multi-scale attention
    multi_attn = MultiScaleSpatialAttention(64, scales=[1, 2], reduction=4)
    multi_params = sum(p.numel() for p in multi_attn.parameters())
    
    with torch.no_grad():
        multi_out = multi_attn(x)
    
    print(f"\nMulti-Scale Attention:")
    print(f"  Parameters: {multi_params:,}")
    print(f"  Scale weights: {multi_attn.scale_weights.detach().numpy()}")
    print(f"  Output change: {torch.norm(multi_out - x).item():.6f}")
    
    print(f"\n🎯 Key Improvements:")
    print(f"  1. Better gamma initialization: {improved_attn.gamma.item():.3f} vs {original_attn.gamma.item():.6f}")
    print(f"  2. Lower reduction ratio: 4 vs 8 (more expressive)")
    print(f"  3. Temperature scaling for sharper attention")
    print(f"  4. Layer normalization for better gradients")
    print(f"  5. Multi-scale processing for different receptive fields")


if __name__ == "__main__":
    test_attention_improvements()