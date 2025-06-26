#!/usr/bin/env python3
"""
Utility functions for palette-based image processing.
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Union
import colorsys


def generate_standard_64_palette() -> np.ndarray:
    """
    Generate a standard 64-color palette using structured color space sampling.
    
    Returns:
        np.ndarray: Shape (64, 3) with RGB values in range [0, 255]
    """
    palette = []
    
    # Start with some standard colors
    standard_colors = [
        [0, 0, 0],        # Black
        [255, 255, 255],  # White
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
    ]
    
    palette.extend(standard_colors)
    
    # Add grayscale values
    gray_levels = np.linspace(32, 224, 8).astype(int)
    for gray in gray_levels:
        palette.append([gray, gray, gray])
    
    # Fill remaining slots with HSV-based sampling
    remaining_slots = 64 - len(palette)
    
    # Sample hues evenly
    hue_count = int(np.sqrt(remaining_slots))
    sat_count = remaining_slots // hue_count
    
    hues = np.linspace(0, 1, hue_count, endpoint=False)
    saturations = np.linspace(0.3, 1.0, sat_count)
    values = [0.5, 0.7, 0.9]
    
    for h in hues:
        for s in saturations:
            for v in values:
                if len(palette) >= 64:
                    break
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                palette.append([int(r * 255), int(g * 255), int(b * 255)])
            if len(palette) >= 64:
                break
        if len(palette) >= 64:
            break
    
    # Ensure we have exactly 64 colors
    while len(palette) < 64:
        # Add random colors if needed
        palette.append([
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256)
        ])
    
    return np.array(palette[:64], dtype=np.uint8)


def posterize_image_to_palette(image: Image.Image, palette: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
    """
    Posterize an image to a specific palette using nearest color matching.
    
    Args:
        image: PIL Image to posterize
        palette: Palette array shape (N, 3) with RGB values [0, 255]
    
    Returns:
        Tuple of (posterized_image, index_map)
        - posterized_image: PIL Image with colors from palette
        - index_map: numpy array (H, W) with palette indices for each pixel
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        # Grayscale image, convert to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    
    h, w, c = img_array.shape
    
    # Reshape image to (H*W, 3)
    pixels = img_array.reshape(-1, 3).astype(np.float32)
    
    # Find nearest palette color for each pixel
    # Compute distances to all palette colors
    distances = np.sum((pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    
    # Find closest palette index for each pixel
    closest_indices = np.argmin(distances, axis=1)
    
    # Map pixels to palette colors
    posterized_pixels = palette[closest_indices]
    
    # Reshape back to image
    posterized_array = posterized_pixels.reshape(h, w, 3)
    index_map = closest_indices.reshape(h, w)
    
    # Convert back to PIL Image
    posterized_image = Image.fromarray(posterized_array.astype(np.uint8))
    
    return posterized_image, index_map


def palette_indices_to_one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert palette indices to one-hot encoding.
    
    Args:
        indices: Array of palette indices (H, W)
        num_classes: Number of palette colors
    
    Returns:
        One-hot encoded array (H, W, num_classes)
    """
    h, w = indices.shape
    one_hot = np.zeros((h, w, num_classes), dtype=np.float32)
    one_hot[np.arange(h)[:, None], np.arange(w), indices] = 1.0
    return one_hot


def one_hot_to_palette_indices(one_hot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoding back to palette indices.
    
    Args:
        one_hot: One-hot encoded array (H, W, num_classes)
    
    Returns:
        Array of palette indices (H, W)
    """
    return np.argmax(one_hot, axis=-1)


def indices_to_rgb(indices: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Convert palette indices to RGB values.
    
    Args:
        indices: Array of palette indices (H, W)
        palette: Palette array (num_colors, 3)
    
    Returns:
        RGB array (H, W, 3)
    """
    return palette[indices]


def tensor_indices_to_rgb(indices: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """
    Convert palette indices tensor to RGB values tensor.
    
    Args:
        indices: Tensor of palette indices (B, H, W)
        palette: Palette tensor (num_colors, 3)
    
    Returns:
        RGB tensor (B, 3, H, W)
    """
    b, h, w = indices.shape
    rgb = palette[indices.long()]  # (B, H, W, 3)
    return rgb.permute(0, 3, 1, 2)  # (B, 3, H, W)


def save_palette_image(palette: np.ndarray, filepath: str, tile_size: int = 32):
    """
    Save a palette as a visual image with color swatches.
    
    Args:
        palette: Palette array (num_colors, 3)
        filepath: Output file path
        tile_size: Size of each color swatch in pixels
    """
    num_colors = len(palette)
    cols = int(np.sqrt(num_colors))
    rows = (num_colors + cols - 1) // cols
    
    # Create image
    img_width = cols * tile_size
    img_height = rows * tile_size
    palette_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    for i, color in enumerate(palette):
        row = i // cols
        col = i % cols
        
        y_start = row * tile_size
        y_end = y_start + tile_size
        x_start = col * tile_size
        x_end = x_start + tile_size
        
        palette_img[y_start:y_end, x_start:x_end] = color
    
    # Save image
    Image.fromarray(palette_img).save(filepath)


if __name__ == "__main__":
    # Test palette generation
    palette = generate_standard_64_palette()
    print(f"Generated palette with {len(palette)} colors")
    print(f"Palette shape: {palette.shape}")
    print(f"First 8 colors: {palette[:8]}")
    
    # Save palette visualization
    save_palette_image(palette, "test_palette.png")
    print("Saved palette visualization to test_palette.png")
    
    # Test posterization
    from PIL import Image
    import os
    
    # Create a test image if none exists
    test_img = Image.new('RGB', (64, 64))
    for y in range(64):
        for x in range(64):
            test_img.putpixel((x, y), (x*4, y*4, (x+y)*2))
    
    posterized, indices = posterize_image_to_palette(test_img, palette)
    
    print(f"Original image size: {test_img.size}")
    print(f"Posterized image size: {posterized.size}")
    print(f"Index map shape: {indices.shape}")
    print(f"Unique palette indices used: {len(np.unique(indices))}")
    
    # Test conversion functions
    one_hot = palette_indices_to_one_hot(indices, 64)
    print(f"One-hot shape: {one_hot.shape}")
    
    recovered_indices = one_hot_to_palette_indices(one_hot)
    print(f"Indices match after one-hot conversion: {np.array_equal(indices, recovered_indices)}")
    
    rgb_recovered = indices_to_rgb(indices, palette)
    print(f"RGB recovery shape: {rgb_recovered.shape}")