#!/usr/bin/env python3
"""
Patch-based inference script for enhanced spatial attention models.

Takes a large image and reconstructs it using trained enhanced spatial attention models.
The model takes 64x64 patches and outputs 16x16 patches.
Uses non-overlapping 16x16 windows for efficient reconstruction.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import sys
import os
import pandas as pd

from enhanced_spatial_attention_models import (
    EnhancedMediumUNetSpatialAttention64x16,
    ConfigurableAttentionUNet64x16
)
from exact_color_models import ExactColorMediumUNetSpatialAttention64x16


def load_model(checkpoint_path: str, model_type: str, device: torch.device, **model_kwargs):
    """Load the trained enhanced spatial attention model from checkpoint."""
    
    if model_type == 'enhanced':
        model = EnhancedMediumUNetSpatialAttention64x16(**model_kwargs)
    elif model_type == 'configurable':
        model = ConfigurableAttentionUNet64x16(**model_kwargs)
    elif model_type == 'exact_color':
        model = ExactColorMediumUNetSpatialAttention64x16(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} model from: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")
    if 'loss' in checkpoint:
        print(f"Best loss: {checkpoint['loss']:.6f}")
    
    return model


def preprocess_patch(patch: np.ndarray, use_exact_colors: bool = False) -> torch.Tensor:
    """Convert numpy patch to tensor and normalize."""
    if use_exact_colors:
        # Keep in [0, 255] range for exact color models
        patch_tensor = torch.from_numpy(patch).float()
    else:
        # Normalize to [0, 1] for standard models
        patch_tensor = torch.from_numpy(patch).float() / 255.0
    
    # Rearrange from HWC to CHW
    patch_tensor = patch_tensor.permute(2, 0, 1)
    # Add batch dimension
    patch_tensor = patch_tensor.unsqueeze(0)
    return patch_tensor


def pad_image_for_reconstruction(image: np.ndarray, input_stride: int = 64) -> tuple:
    """
    Pad image to make dimensions divisible by input_stride.
    
    Args:
        image: Input image as numpy array (H, W, C)
        input_stride: Size of input stride (64x64) for 1/4 resolution output
    
    Returns:
        Tuple of (padded_image, padding_info)
    """
    height, width, channels = image.shape
    
    # Calculate padding needed to make dimensions divisible by input_stride
    pad_height = (input_stride - height % input_stride) % input_stride
    pad_width = (input_stride - width % input_stride) % input_stride
    
    # Pad the image
    padded_image = np.pad(image, 
                         ((0, pad_height), (0, pad_width), (0, 0)), 
                         mode='reflect')
    
    padding_info = {
        'pad_height': pad_height,
        'pad_width': pad_width,
        'original_shape': (height, width, channels)
    }
    
    return padded_image, padding_info


def tile_based_inference(image: np.ndarray, model: torch.nn.Module, device: torch.device, 
                        patch_size: int = 16, window_size: int = 64, 
                        use_exact_colors: bool = False) -> tuple:
    """
    Perform tile-based inference on the input image.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model: Trained enhanced spatial attention model
        device: PyTorch device
        patch_size: Size of output patches (16x16)
        window_size: Size of input window (64x64)
        use_exact_colors: Whether to use exact color processing
    
    Returns:
        Tuple of (mean_output, variance_output) both at 1/4 resolution of input
    """
    # Input stride is the window size for 1/4 resolution output
    input_stride = window_size
    
    # Pad image to handle edge cases and make dimensions divisible by input stride
    padded_image, padding_info = pad_image_for_reconstruction(image, input_stride)
    padded_height, padded_width, channels = padded_image.shape
    
    # Calculate number of tiles based on input stride
    num_tiles_y = padded_height // input_stride
    num_tiles_x = padded_width // input_stride
    
    # Initialize output arrays (1/4 resolution)
    output_height = padded_height // 4
    output_width = padded_width // 4
    mean_output = np.zeros((output_height, output_width, channels), dtype=np.float32)
    variance_output = np.zeros((output_height, output_width, channels), dtype=np.float32)
    
    print(f"Input image shape: {image.shape}")
    print(f"Padded image shape: {padded_image.shape}")
    print(f"Output image shape: {mean_output.shape}")
    print(f"Number of tiles: {num_tiles_y} x {num_tiles_x} = {num_tiles_y * num_tiles_x}")
    print(f"Patch size: {patch_size}x{patch_size}, Window size: {window_size}x{window_size}")
    print(f"Input stride: {input_stride}x{input_stride}")
    print(f"Using exact colors: {use_exact_colors}")
    
    with torch.no_grad():
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                # Calculate input tile coordinates (stride by window_size)
                input_y_start = tile_y * input_stride
                input_x_start = tile_x * input_stride
                
                # Extract 64x64 input patch
                input_patch = padded_image[input_y_start:input_y_start + window_size,
                                         input_x_start:input_x_start + window_size, :]
                
                # Ensure patch is the right size
                if input_patch.shape[:2] != (window_size, window_size):
                    print(f"Warning: Patch at tile ({tile_y}, {tile_x}) has wrong size: {input_patch.shape}")
                    continue
                
                # Preprocess patch
                patch_tensor = preprocess_patch(input_patch, use_exact_colors).to(device)
                
                # Get model prediction (16x16 patch only)
                mu = model(patch_tensor)
                
                # Convert predictions back to numpy
                # mu shape: (1, 3, 16, 16)
                patch_mean = mu.squeeze(0).cpu().numpy().transpose(1, 2, 0)      # (16, 16, 3)
                
                # Store in output arrays (coordinates in output space)
                output_y_start = tile_y * patch_size
                output_x_start = tile_x * patch_size
                output_y_end = output_y_start + patch_size
                output_x_end = output_x_start + patch_size
                mean_output[output_y_start:output_y_end, output_x_start:output_x_end, :] = patch_mean
                
                # Progress indicator
                tiles_done = tile_y * num_tiles_x + tile_x + 1
                total_tiles = num_tiles_y * num_tiles_x
                if tiles_done % 10 == 0 or tiles_done == total_tiles:
                    progress = tiles_done / total_tiles * 100
                    print(f"Progress: {progress:.1f}% ({tiles_done}/{total_tiles} tiles)")
    
    # Remove padding to get back to original dimensions (1/4 scale)
    orig_height, orig_width, _ = padding_info['original_shape']
    orig_output_height = orig_height // 4
    orig_output_width = orig_width // 4
    mean_output = mean_output[:orig_output_height, :orig_output_width, :]
    variance_output = variance_output[:orig_output_height, :orig_output_width, :]
    
    return mean_output, variance_output


def batch_tile_inference(image: np.ndarray, model: torch.nn.Module, device: torch.device, 
                        patch_size: int = 16, window_size: int = 64, batch_size: int = 16,
                        use_exact_colors: bool = False) -> tuple:
    """
    Perform batched tile-based inference for better GPU utilization.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model: Trained enhanced spatial attention model
        device: PyTorch device
        patch_size: Size of output patches (16x16)
        window_size: Size of input window (64x64)
        batch_size: Number of tiles to process in parallel
        use_exact_colors: Whether to use exact color processing
    
    Returns:
        Tuple of (mean_output, variance_output) both at 1/4 resolution of input
    """
    # Input stride is the window size for 1/4 resolution output
    input_stride = window_size
    
    # Pad image to handle edge cases and make dimensions divisible by input stride
    padded_image, padding_info = pad_image_for_reconstruction(image, input_stride)
    padded_height, padded_width, channels = padded_image.shape
    
    # Calculate number of tiles based on input stride
    num_tiles_y = padded_height // input_stride
    num_tiles_x = padded_width // input_stride
    total_tiles = num_tiles_y * num_tiles_x
    
    # Initialize output arrays (1/4 resolution)
    output_height = padded_height // 4
    output_width = padded_width // 4
    mean_output = np.zeros((output_height, output_width, channels), dtype=np.float32)
    variance_output = np.zeros((output_height, output_width, channels), dtype=np.float32)
    
    print(f"Batched Inference:")
    print(f"Input image shape: {image.shape}")
    print(f"Padded image shape: {padded_image.shape}")
    print(f"Output image shape: {mean_output.shape}")
    print(f"Number of tiles: {num_tiles_y} x {num_tiles_x} = {total_tiles}")
    print(f"Patch size: {patch_size}x{patch_size}, Window size: {window_size}x{window_size}")
    print(f"Input stride: {input_stride}x{input_stride}")
    print(f"Batch size: {batch_size}")
    print(f"Using exact colors: {use_exact_colors}")
    
    # Collect all tiles and coordinates
    tiles = []
    coordinates = []
    
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            # Calculate input tile coordinates (stride by window_size)
            input_y_start = tile_y * input_stride
            input_x_start = tile_x * input_stride
            
            # Extract 64x64 input patch
            input_patch = padded_image[input_y_start:input_y_start + window_size,
                                     input_x_start:input_x_start + window_size, :]
            
            if input_patch.shape[:2] == (window_size, window_size):
                tiles.append(input_patch)
                coordinates.append((tile_y, tile_x))
    
    # Process tiles in batches
    with torch.no_grad():
        for batch_start in range(0, len(tiles), batch_size):
            batch_end = min(batch_start + batch_size, len(tiles))
            batch_tiles = tiles[batch_start:batch_end]
            batch_coords = coordinates[batch_start:batch_end]
            
            # Preprocess batch
            batch_tensor = torch.stack([
                preprocess_patch(tile, use_exact_colors).squeeze(0) for tile in batch_tiles
            ]).to(device)
            
            # Get model predictions
            mu_batch = model(batch_tensor)  # Shape: (batch_size, 3, 16, 16)
            
            # Store results
            for idx, (tile_y, tile_x) in enumerate(batch_coords):
                patch_mean = mu_batch[idx].cpu().numpy().transpose(1, 2, 0)      # (16, 16, 3)
                
                # Store in output arrays (coordinates in output space)
                output_y_start = tile_y * patch_size
                output_x_start = tile_x * patch_size
                output_y_end = output_y_start + patch_size
                output_x_end = output_x_start + patch_size
                mean_output[output_y_start:output_y_end, output_x_start:output_x_end, :] = patch_mean
            
            # Progress indicator
            progress = batch_end / len(tiles) * 100
            print(f"Progress: {progress:.1f}% ({batch_end}/{len(tiles)} tiles)")
    
    # Remove padding to get back to original dimensions (1/4 scale)
    orig_height, orig_width, _ = padding_info['original_shape']
    orig_output_height = orig_height // 4
    orig_output_width = orig_width // 4
    mean_output = mean_output[:orig_output_height, :orig_output_width, :]
    variance_output = variance_output[:orig_output_height, :orig_output_width, :]
    
    return mean_output, variance_output


def postprocess_output(output: np.ndarray, use_exact_colors: bool = False) -> np.ndarray:
    """Convert output tensor to displayable image."""
    if use_exact_colors:
        # Output is already in [0, 255] range
        output = np.clip(output, 0, 255).astype(np.uint8)
    else:
        # Clip values to [0, 1] range and convert to [0, 255]
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
    return output


def main():
    parser = argparse.ArgumentParser(description='Patch-based inference for enhanced spatial attention models')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='enhanced',
                       choices=['enhanced', 'configurable', 'exact_color'],
                       help='Type of model to load')
    parser.add_argument('--output', type=str, help='Output image path (default: input_reconstructed.png)')
    
    # Model configuration (should match training configuration)
    parser.add_argument('--attention-type', type=str, default='improved',
                       choices=['improved', 'positional', 'multiscale'],
                       help='Type of spatial attention mechanism')
    parser.add_argument('--attention-strength', type=float, default=0.2,
                       help='Attention strength parameter')
    parser.add_argument('--attention-temperature', type=float, default=0.7,
                       help='Attention temperature parameter')
    parser.add_argument('--attention-reduction', type=int, default=4,
                       help='Attention reduction factor')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--use-transpose', action='store_true', default=True,
                       help='Use transposed convolution for upsampling')
    parser.add_argument('--final-activation', type=str, default='sigmoid',
                       choices=['sigmoid', 'tanh', 'relu', 'none'],
                       help='Final activation function')
    
    # Inference parameters
    parser.add_argument('--patch-size', type=int, default=16, help='Output patch size')
    parser.add_argument('--window-size', type=int, default=64, help='Input window size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--use-batched', action='store_true', help='Use batched inference (faster)')
    parser.add_argument('--use-exact-colors', action='store_true',
                       help='Use exact color processing (for exact_color models)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load input image
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found")
        return
    
    image = Image.open(args.input_image).convert('RGB')
    image_array = np.array(image)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint '{args.model}' not found")
        return
    
    # Prepare model kwargs
    model_kwargs = {
        'dropout': args.dropout,
        'use_transpose': args.use_transpose,
        'final_activation': args.final_activation,
    }
    
    if args.model_type in ['enhanced', 'configurable']:
        model_kwargs.update({
            'attention_type': args.attention_type,
            'attention_strength': args.attention_strength,
            'attention_temperature': args.attention_temperature,
            'attention_reduction': args.attention_reduction,
        })
    
    print(f"Loading {args.model_type} model...")
    model = load_model(args.model, args.model_type, device, **model_kwargs)
    
    # Get attention info if available
    if hasattr(model, 'get_attention_info'):
        attention_info = model.get_attention_info()
        print(f"Attention parameters: {attention_info['parameters']:,}")
        print(f"Attention config: {attention_info['config']}")
    
    # Perform inference
    print("Starting patch-based inference...")
    if args.use_batched:
        mean_output, variance_output = batch_tile_inference(
            image_array, model, device, 
            patch_size=args.patch_size, 
            window_size=args.window_size,
            batch_size=args.batch_size,
            use_exact_colors=args.use_exact_colors)
    else:
        mean_output, variance_output = tile_based_inference(
            image_array, model, device, 
            patch_size=args.patch_size, 
            window_size=args.window_size,
            use_exact_colors=args.use_exact_colors)
    
    # Postprocess output
    output_image = postprocess_output(mean_output, args.use_exact_colors)
    
    # Save output
    if args.output is None:
        input_path = Path(args.input_image)
        output_path = input_path.parent / f"{input_path.stem}_reconstructed{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    Image.fromarray(output_image).save(output_path)
    print(f"Reconstructed image saved to: {output_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Input resolution: {image_array.shape[1]}x{image_array.shape[0]}")
    print(f"Output resolution: {output_image.shape[1]}x{output_image.shape[0]}")
    print(f"Model type: {args.model_type}")
    print(f"Attention type: {args.attention_type}")
    print(f"Attention strength: {args.attention_strength}")
    print(f"Attention temperature: {args.attention_temperature}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Window size: {args.window_size}x{args.window_size}")


if __name__ == "__main__":
    main()