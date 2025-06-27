#!/usr/bin/env python3
"""
Enhanced U-Net models with improved spatial attention mechanisms.

These models address the issues in the original spatial attention implementation:
1. Better gamma initialization (0.1-0.2 instead of 0.0)
2. Lower reduction ratios for more expressiveness
3. Temperature control for attention sharpness
4. Multi-scale attention options
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import existing components
from medium_unet import (
    GroupNormAct, ResidualBlock, SEBlock, EncoderBlock, DecoderBlock,
    count_parameters, mse_loss
)
from improved_spatial_attention import (
    ImprovedSpatialSelfAttention, 
    PositionalSpatialAttention,
    MultiScaleSpatialAttention
)


class EnhancedResidualBlockWithAttention(nn.Module):
    """Enhanced residual block with improved spatial attention."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, 
                 attention_type: str = 'improved', attention_strength: float = 0.2,
                 temperature: float = 1.0, reduction: int = 4):
        super(EnhancedResidualBlockWithAttention, self).__init__()
        
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
        
        # Enhanced spatial attention options
        if attention_type == 'improved':
            self.attention = ImprovedSpatialSelfAttention(
                out_channels, reduction=reduction, temperature=temperature, 
                init_gamma=attention_strength
            )
        elif attention_type == 'positional':
            self.attention = PositionalSpatialAttention(
                out_channels, reduction=reduction, temperature=temperature
            )
        elif attention_type == 'multiscale':
            self.attention = MultiScaleSpatialAttention(
                out_channels, scales=[1, 2], reduction=reduction, temperature=temperature
            )
        else:
            self.attention = None
    
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
        
        # Apply enhanced spatial attention
        if self.attention is not None:
            out = self.attention(out)
        
        return out


class EnhancedMediumUNetSpatialAttention64x16(nn.Module):
    """
    Enhanced Medium U-Net with improved spatial attention for 64x64 → 16x16.
    
    Key improvements:
    - Better attention initialization (gamma=0.1-0.2 instead of 0.0)
    - Lower reduction ratios (4 instead of 8)
    - Temperature control for attention sharpness
    - Configurable attention strength
    - Multiple attention types (improved, positional, multi-scale)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid', 
                 attention_type: str = 'improved',
                 attention_strength: float = 0.2,
                 attention_temperature: float = 0.7,
                 attention_reduction: int = 4):
        super(EnhancedMediumUNetSpatialAttention64x16, self).__init__()
        
        self.final_activation = final_activation
        self.attention_type = attention_type
        
        # Encoder path - 3 levels for 64x64 input (reuse existing EncoderBlock)
        # 64x64 → 32x32
        self.enc1 = EncoderBlock(in_channels, 32, dropout * 0.5)
        # 32x32 → 16x16  
        self.enc2 = EncoderBlock(32, 64, dropout * 0.7)
        # 16x16 → 8x8
        self.enc3 = EncoderBlock(64, 128, dropout * 0.8)
        
        # Enhanced bottleneck at 8x8 with improved spatial attention
        self.bottleneck = nn.Sequential(
            EnhancedResidualBlockWithAttention(
                128, 128, dropout, 
                attention_type=attention_type,
                attention_strength=attention_strength,
                temperature=attention_temperature,
                reduction=attention_reduction
            ),
            EnhancedResidualBlockWithAttention(
                128, 128, dropout,
                attention_type=attention_type, 
                attention_strength=attention_strength * 0.8,  # Slightly less for second block
                temperature=attention_temperature,
                reduction=attention_reduction
            ),
            SEBlock(128)
        )
        
        # Decoder path with spatial attention - to exact 16x16 output
        # 8x8 → 16x16 (with skip from enc3: 128 channels)
        self.dec3 = DecoderBlock(128, 128, 64, dropout, True, use_transpose)
        
        # Enhanced spatial attention in decoder features before final conv
        if attention_type == 'improved':
            self.decoder_attention = ImprovedSpatialSelfAttention(
                64, reduction=attention_reduction, temperature=attention_temperature,
                init_gamma=attention_strength
            )
        elif attention_type == 'positional':
            self.decoder_attention = PositionalSpatialAttention(
                64, reduction=attention_reduction, temperature=attention_temperature
            )
        elif attention_type == 'multiscale':
            self.decoder_attention = MultiScaleSpatialAttention(
                64, scales=[1, 2], reduction=attention_reduction, temperature=attention_temperature
            )
        else:
            self.decoder_attention = None
        
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
        
        # Enhanced bottleneck with improved spatial attention
        x = self.bottleneck(x3)     # 8x8x128
        
        # Decoder path with attention-gated skip connections
        dec_out = self.dec3(x, skip3)     # 8x8 → 16x16, concat with skip3
        
        # Apply enhanced spatial attention to decoder features
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


class ConfigurableAttentionUNet64x16(nn.Module):
    """
    U-Net with highly configurable attention settings for experimentation.
    
    This model allows easy tuning of attention parameters to find optimal settings.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 dropout: float = 0.1, use_transpose: bool = True,
                 final_activation: str = 'sigmoid',
                 
                 # Attention configuration
                 use_attention: bool = True,
                 attention_type: str = 'improved',
                 attention_locations: list = ['bottleneck', 'decoder'],
                 attention_strength: float = 0.2,
                 attention_temperature: float = 0.7,
                 attention_reduction: int = 4,
                 
                 # Multi-scale specific
                 multiscale_scales: list = [1, 2],
                 
                 # Positional specific  
                 positional_max_size: int = 32):
        
        super(ConfigurableAttentionUNet64x16, self).__init__()
        
        self.final_activation = final_activation
        self.use_attention = use_attention
        self.attention_locations = attention_locations
        
        # Store attention config for logging
        self.attention_config = {
            'type': attention_type,
            'strength': attention_strength,
            'temperature': attention_temperature,
            'reduction': attention_reduction,
            'locations': attention_locations
        }
        
        # Encoder path
        self.enc1 = EncoderBlock(in_channels, 32, dropout * 0.5)
        self.enc2 = EncoderBlock(32, 64, dropout * 0.7)
        self.enc3 = EncoderBlock(64, 128, dropout * 0.8)
        
        # Configurable bottleneck
        bottleneck_layers = []
        if 'bottleneck' in attention_locations and use_attention:
            bottleneck_layers.extend([
                EnhancedResidualBlockWithAttention(
                    128, 128, dropout,
                    attention_type=attention_type,
                    attention_strength=attention_strength,
                    temperature=attention_temperature,
                    reduction=attention_reduction
                ),
                EnhancedResidualBlockWithAttention(
                    128, 128, dropout,
                    attention_type=attention_type,
                    attention_strength=attention_strength * 0.8,
                    temperature=attention_temperature,
                    reduction=attention_reduction
                )
            ])
        else:
            bottleneck_layers.extend([
                ResidualBlock(128, 128, dropout),
                ResidualBlock(128, 128, dropout)
            ])
        
        bottleneck_layers.append(SEBlock(128))
        self.bottleneck = nn.Sequential(*bottleneck_layers)
        
        # Decoder
        self.dec3 = DecoderBlock(128, 128, 64, dropout, True, use_transpose)
        
        # Configurable decoder attention
        if 'decoder' in attention_locations and use_attention:
            if attention_type == 'improved':
                self.decoder_attention = ImprovedSpatialSelfAttention(
                    64, reduction=attention_reduction, temperature=attention_temperature,
                    init_gamma=attention_strength
                )
            elif attention_type == 'positional':
                self.decoder_attention = PositionalSpatialAttention(
                    64, reduction=attention_reduction, max_size=positional_max_size,
                    temperature=attention_temperature
                )
            elif attention_type == 'multiscale':
                self.decoder_attention = MultiScaleSpatialAttention(
                    64, scales=multiscale_scales, reduction=attention_reduction,
                    temperature=attention_temperature
                )
        else:
            self.decoder_attention = None
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            GroupNormAct(32),
            nn.Conv2d(32, out_channels, 1)
        )
    
    def forward(self, x):
        # Encoder
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder
        dec_out = self.dec3(x, skip3)
        
        # Optional decoder attention
        if self.decoder_attention is not None:
            dec_out = self.decoder_attention(dec_out)
        
        # Final output
        mu = self.final_conv(dec_out)
        
        # Activation
        if self.final_activation == 'sigmoid':
            mu = torch.sigmoid(mu)
        elif self.final_activation == 'tanh':
            mu = torch.tanh(mu)
        elif self.final_activation == 'relu':
            mu = torch.relu(mu)
        
        return mu
    
    def get_attention_info(self):
        """Return information about attention configuration."""
        total_attention_params = 0
        attention_modules = []
        
        # Count attention parameters
        for name, module in self.named_modules():
            if any(attn_type in module.__class__.__name__ for attn_type in 
                   ['SpatialSelfAttention', 'PositionalSpatialAttention', 'MultiScaleSpatialAttention']):
                attention_modules.append(name)
                total_attention_params += sum(p.numel() for p in module.parameters())
        
        return {
            'config': self.attention_config,
            'modules': attention_modules,
            'parameters': total_attention_params
        }


def test_enhanced_models():
    """Test the enhanced attention models."""
    
    print("Testing Enhanced Spatial Attention Models")
    print("=" * 50)
    
    x = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 16, 16)
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    print()
    
    # Test different attention types
    attention_types = ['improved', 'positional', 'multiscale']
    attention_strengths = [0.1, 0.2, 0.3]
    temperatures = [0.5, 0.7, 1.0]
    
    results = []
    
    for attn_type in attention_types:
        for strength in [0.2]:  # Test just one strength for brevity
            for temp in [0.7]:  # Test just one temperature for brevity
                
                print(f"Testing {attn_type} attention (strength={strength}, temp={temp}):")
                
                model = EnhancedMediumUNetSpatialAttention64x16(
                    attention_type=attn_type,
                    attention_strength=strength,
                    attention_temperature=temp,
                    attention_reduction=4
                )
                
                params = count_parameters(model)
                
                with torch.no_grad():
                    output = model(x)
                    loss = mse_loss(output, target)
                
                print(f"  Parameters: {params:,}")
                print(f"  Output shape: {output.shape}")
                print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"  Loss: {loss:.3f}")
                
                results.append({
                    'type': attn_type,
                    'strength': strength,
                    'temperature': temp,
                    'parameters': params,
                    'loss': loss.item()
                })
                print()
    
    # Test configurable model
    print("Testing Configurable Attention Model:")
    configurable_model = ConfigurableAttentionUNet64x16(
        attention_type='improved',
        attention_strength=0.25,
        attention_temperature=0.6,
        attention_locations=['bottleneck', 'decoder']
    )
    
    config_params = count_parameters(configurable_model)
    attention_info = configurable_model.get_attention_info()
    
    with torch.no_grad():
        config_output = configurable_model(x)
        config_loss = mse_loss(config_output, target)
    
    print(f"  Parameters: {config_params:,}")
    print(f"  Attention parameters: {attention_info['parameters']:,}")
    print(f"  Attention modules: {attention_info['modules']}")
    print(f"  Loss: {config_loss:.3f}")
    print()
    
    # Compare with original model
    print("Comparison with Original Model:")
    from medium_unet import MediumUNetSpatialAttention64x16
    original_model = MediumUNetSpatialAttention64x16()
    original_params = count_parameters(original_model)
    
    with torch.no_grad():
        original_output = original_model(x)
        original_loss = mse_loss(original_output, target)
    
    print(f"  Original Parameters: {original_params:,}")
    print(f"  Original Loss: {original_loss:.3f}")
    print(f"  Enhanced vs Original: {config_loss:.3f} vs {original_loss:.3f}")
    
    # Show recommended settings
    print("\n🎯 Recommended Settings for Pixel Art:")
    print("  - attention_type: 'improved'")
    print("  - attention_strength: 0.2-0.3")
    print("  - attention_temperature: 0.6-0.8")
    print("  - attention_reduction: 4")
    print("  - attention_locations: ['bottleneck', 'decoder']")


if __name__ == "__main__":
    test_enhanced_models()