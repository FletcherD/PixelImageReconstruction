#!/usr/bin/env python3
"""
Exact color transforms that preserve precise RGB values for pixel art.

These transforms avoid the precision loss that occurs with standard torchvision
transforms.ToTensor() which normalizes [0,255] to [0,1] by dividing by 255.
Instead, we keep RGB values as exact integers in [0,255] range throughout the pipeline.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union


class ExactRGBToTensor:
    """
    Convert PIL Image to tensor while preserving exact RGB integer values.
    
    Unlike transforms.ToTensor() which normalizes to [0,1], this keeps
    RGB values in the original [0,255] integer range for exact color preservation.
    """
    
    def __call__(self, img: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Convert PIL Image or numpy array to tensor preserving exact RGB values.
        
        Args:
            img: PIL Image in RGB mode or numpy array (H, W, C)
            
        Returns:
            torch.Tensor: (C, H, W) tensor with values in [0, 255] range
        """
        if isinstance(img, Image.Image):
            # Convert PIL Image to numpy array
            img_array = np.array(img, dtype=np.uint8)
        else:
            # Already numpy array
            img_array = img.astype(np.uint8)
        
        # Convert to tensor and rearrange from HWC to CHW
        tensor = torch.from_numpy(img_array).float()
        
        # Rearrange dimensions: (H, W, C) -> (C, H, W)
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)
        
        return tensor


class ExactRGBToImage:
    """
    Convert tensor with exact RGB values back to PIL Image.
    
    Expects tensor with values in [0, 255] range.
    """
    
    def __call__(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image preserving exact RGB values.
        
        Args:
            tensor: (C, H, W) tensor with values in [0, 255] range
            
        Returns:
            PIL.Image: RGB image with exact integer values
        """
        # Rearrange from CHW to HWC
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # Clamp to valid range and convert to uint8
        tensor = torch.clamp(tensor, 0, 255)
        img_array = tensor.detach().cpu().numpy().astype(np.uint8)
        
        # Convert to PIL Image
        return Image.fromarray(img_array, mode='RGB')


def preprocess_exact_rgb(img: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    """
    Preprocess image for exact RGB preservation.
    
    Args:
        img: PIL Image or numpy array
        
    Returns:
        torch.Tensor: Preprocessed tensor with exact RGB values in [0, 255]
    """
    transform = ExactRGBToTensor()
    return transform(img)


def postprocess_exact_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Postprocess tensor back to exact RGB numpy array.
    
    Args:
        tensor: (C, H, W) or (H, W, C) tensor with values in [0, 255] range
        
    Returns:
        np.ndarray: RGB array with exact integer values
    """
    # Handle both CHW and HWC formats
    if tensor.dim() == 3:
        if tensor.shape[0] == 3:  # CHW format
            tensor = tensor.permute(1, 2, 0)  # Convert to HWC
    
    # Clamp and convert to uint8
    tensor = torch.clamp(tensor, 0, 255)
    return tensor.detach().cpu().numpy().astype(np.uint8)


class ExactColorCompose:
    """
    Compose multiple exact color transforms.
    Similar to torchvision.transforms.Compose but for exact RGB preservation.
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img


def test_exact_color_preservation():
    """Test that exact RGB values are preserved through the transform pipeline."""
    
    print("Testing exact color preservation...")
    
    # Create test image with problematic RGB values that lose precision with /255 normalization
    test_colors = [
        [128, 200, 64],   # 128/255 = 0.5020392... -> 128.01960 when *255
        [85, 170, 42],    # 85/255 = 0.3333333... -> 84.99999 when *255
        [127, 63, 191],   # Values that don't divide evenly by 255
        [1, 2, 254]       # Edge cases
    ]
    
    # Create a small test image
    test_array = np.array([[test_colors[0], test_colors[1]], 
                          [test_colors[2], test_colors[3]]], dtype=np.uint8)
    test_image = Image.fromarray(test_array, mode='RGB')
    
    print(f"Original colors: {test_colors}")
    
    # Test exact RGB transform
    exact_transform = ExactRGBToTensor()
    tensor = exact_transform(test_image)
    
    print(f"\nExact RGB Transform:")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor value range: [{tensor.min():.1f}, {tensor.max():.1f}]")
    
    # Convert back
    back_transform = ExactRGBToImage()
    recovered_image = back_transform(tensor)
    recovered_array = np.array(recovered_image)
    
    print(f"Recovered colors:")
    exact_matches = 0
    for i in range(2):
        for j in range(2):
            recovered_color = recovered_array[i, j].tolist()
            original_color = test_colors[i * 2 + j]
            match = recovered_color == original_color
            if match:
                exact_matches += 1
            print(f"  {original_color} -> {recovered_color} {'✓' if match else '✗'}")
    
    # Compare with standard ToTensor
    print("\nStandard ToTensor() + model simulation:")
    from torchvision import transforms
    standard_transform = transforms.ToTensor()
    standard_tensor = standard_transform(test_image)
    
    print(f"After ToTensor normalization [0,1] range:")
    print(f"Value range: [{standard_tensor.min():.6f}, {standard_tensor.max():.6f}]")
    
    # Simulate what happens in model: values stay in [0,1] but with float precision
    # Then convert back to [0,255] as done in inference
    standard_back = (standard_tensor * 255.0).round().clamp(0, 255).byte()
    standard_back = standard_back.permute(1, 2, 0).numpy()
    
    print(f"After *255 denormalization:")
    standard_matches = 0
    for i in range(2):
        for j in range(2):
            recovered_color = standard_back[i, j].tolist()
            original_color = test_colors[i * 2 + j]
            match = recovered_color == original_color
            if match:
                standard_matches += 1
            print(f"  {original_color} -> {recovered_color} {'✓' if match else '✗'}")
    
    # Test more problematic values that definitely lose precision
    print(f"\nTesting precision loss with non-255-divisible values:")
    problematic_values = [43, 86, 129, 172, 215]  # These don't divide evenly by 255
    
    for val in problematic_values:
        # Simulate the normalization and denormalization
        normalized = val / 255.0
        denormalized = int(normalized * 255.0)
        precision_lost = val != denormalized
        print(f"  {val} -> {normalized:.6f} -> {denormalized} {'✗' if precision_lost else '✓'}")
    
    print(f"\nSummary:")
    print(f"Exact RGB Transform: {exact_matches}/4 colors preserved exactly")
    print(f"Standard ToTensor: {standard_matches}/4 colors preserved exactly")


if __name__ == "__main__":
    test_exact_color_preservation()