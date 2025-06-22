#!/usr/bin/env python3
"""
Scale and offset detection model for pixel grid transformation estimation.

Takes a 128x128 input image and outputs 4 scalars representing:
- scale_x: Horizontal scaling factor of the pixel grid
- scale_y: Vertical scaling factor of the pixel grid  
- offset_x: Sub-pixel horizontal grid alignment offset
- offset_y: Sub-pixel vertical grid alignment offset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ScaleOffsetDetector(nn.Module):
    """Lightweight CNN for detecting scale and offset transformations in pixel grids."""
    
    def __init__(self, input_size: int = 128, dropout_rate: float = 0.2):
        """
        Args:
            input_size: Input image size (assumes square images)
            dropout_rate: Dropout rate in the regression head
        """
        super().__init__()
        self.input_size = input_size
        
        # Encoder backbone - progressive downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 4x4x512 -> 1x1x512
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 4)  # Output: [scale_x, scale_y, offset_x, offset_y]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, 4) containing [scale_x, scale_y, offset_x, offset_y]
        """
        # Encoder backbone
        x = self.conv1(x)  # (B, 32, 64, 64)
        x = self.conv2(x)  # (B, 64, 32, 32)
        x = self.conv3(x)  # (B, 128, 16, 16)
        x = self.conv4(x)  # (B, 256, 8, 8)
        x = self.conv5(x)  # (B, 512, 4, 4)
        
        # Global pooling
        x = self.global_pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        
        # Regression head
        x = self.regression_head(x)  # (B, 4)
        
        return x
    
    def predict_transform_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict transformation parameters and return them as separate scale and offset tensors.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of (scale, offset) tensors:
            - scale: (batch_size, 2) containing [scale_x, scale_y]
            - offset: (batch_size, 2) containing [offset_x, offset_y]
        """
        output = self.forward(x)  # (B, 4)
        scale = output[:, :2]     # (B, 2) - [scale_x, scale_y]
        offset = output[:, 2:]    # (B, 2) - [offset_x, offset_y]
        return scale, offset


class ScaleOffsetLoss(nn.Module):
    """Combined loss for scale and offset prediction."""
    
    def __init__(self, scale_weight: float = 1.0, offset_weight: float = 1.0):
        """
        Args:
            scale_weight: Weight for scale loss component
            offset_weight: Weight for offset loss component
        """
        super().__init__()
        self.scale_weight = scale_weight
        self.offset_weight = offset_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined scale and offset loss.
        
        Args:
            pred: Predicted parameters (batch_size, 4) - [scale_x, scale_y, offset_x, offset_y]
            target: Target parameters (batch_size, 4) - [scale_x, scale_y, offset_x, offset_y]
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        pred_scale = pred[:, :2]    # [scale_x, scale_y]
        pred_offset = pred[:, 2:]   # [offset_x, offset_y]
        
        target_scale = target[:, :2]    # [scale_x, scale_y]
        target_offset = target[:, 2:]   # [offset_x, offset_y]
        
        # Compute individual losses
        scale_loss = self.mse_loss(pred_scale, target_scale)
        offset_loss = self.mse_loss(pred_offset, target_offset)
        
        # Combined loss
        total_loss = self.scale_weight * scale_loss + self.offset_weight * offset_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'scale_loss': scale_loss.item(),
            'offset_loss': offset_loss.item()
        }
        
        return total_loss, loss_dict


def create_scale_offset_model(input_size: int = 128, dropout_rate: float = 0.2) -> ScaleOffsetDetector:
    """
    Create a scale/offset detection model.
    
    Args:
        input_size: Input image size
        dropout_rate: Dropout rate in regression head
        
    Returns:
        ScaleOffsetDetector model
    """
    return ScaleOffsetDetector(input_size=input_size, dropout_rate=dropout_rate)


if __name__ == "__main__":
    # Test the model
    model = create_scale_offset_model()
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 128, 128)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output (first sample): {output[0].detach().numpy()}")
    
    # Test prediction method
    scale, offset = model.predict_transform_params(input_tensor)
    print(f"Scale shape: {scale.shape}, Offset shape: {offset.shape}")
    print(f"Scale (first sample): {scale[0].detach().numpy()}")
    print(f"Offset (first sample): {offset[0].detach().numpy()}")
    
    # Test loss
    target = torch.randn(batch_size, 4)
    loss_fn = ScaleOffsetLoss()
    total_loss, loss_dict = loss_fn(output, target)
    print(f"Loss: {loss_dict}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")