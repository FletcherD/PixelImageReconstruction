#!/usr/bin/env python3
"""
Patch-based inference script for image reconstruction using MediumUNet.

Takes a large image and reconstructs it using the trained MediumUNet model.
The model takes 128x128 patches and outputs 32x32 patches.
Uses non-overlapping 32x32 windows for efficient reconstruction.
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

from medium_unet import MediumUNet


def enable_dropout(model):
    """Enable dropout layers during inference for Monte Carlo dropout."""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def disable_dropout(model):
    """Disable dropout layers (standard inference)."""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.eval()


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained MediumUNet model from checkpoint."""
    model = MediumUNet()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """Convert numpy patch to tensor and normalize to [0, 1]."""
    # Convert to tensor and normalize
    patch_tensor = torch.from_numpy(patch).float() / 255.0
    # Rearrange from HWC to CHW
    patch_tensor = patch_tensor.permute(2, 0, 1)
    # Add batch dimension
    patch_tensor = patch_tensor.unsqueeze(0)
    return patch_tensor


def pad_image_for_reconstruction(image: np.ndarray, input_stride: int = 128) -> tuple:
    """
    Pad image to make dimensions divisible by input_stride.
    
    Args:
        image: Input image as numpy array (H, W, C)
        input_stride: Size of input stride (128x128) for 1/4 resolution output
    
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


def extract_input_patch(image: np.ndarray, i: int, j: int, 
                       patch_size: int = 32, window_size: int = 128) -> np.ndarray:
    """
    Extract 128x128 input patch centered on 32x32 output region.
    
    Args:
        image: Input image
        i, j: Top-left coordinates of 32x32 output region
        patch_size: Size of output patch (32)
        window_size: Size of input window (128)
    
    Returns:
        128x128 input patch
    """
    # Calculate center of 32x32 region
    center_y = i + patch_size // 2
    center_x = j + patch_size // 2
    
    # Calculate 128x128 window coordinates
    half_window = window_size // 2
    y_start = center_y - half_window
    y_end = center_y + half_window
    x_start = center_x - half_window  
    x_end = center_x + half_window
    
    # Handle edge cases with padding
    height, width = image.shape[:2]
    
    # Create padded coordinates
    pad_top = max(0, -y_start)
    pad_bottom = max(0, y_end - height)
    pad_left = max(0, -x_start)
    pad_right = max(0, x_end - width)
    
    # Adjust coordinates to valid range
    y_start = max(0, y_start)
    y_end = min(height, y_end)
    x_start = max(0, x_start)
    x_end = min(width, x_end)
    
    # Extract patch
    patch = image[y_start:y_end, x_start:x_end, :]
    
    # Apply padding if needed
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        patch = np.pad(patch, 
                      ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                      mode='reflect')
    
    return patch


def mc_dropout_inference_single_patch(model: torch.nn.Module, patch_tensor: torch.Tensor, 
                                    n_samples: int = 10) -> tuple:
    """
    Perform Monte Carlo dropout inference on a single patch.
    Note: With heteroscedastic models, this samples the mu outputs to estimate epistemic uncertainty,
    while the model's sigma_sq represents aleatoric uncertainty.
    
    Args:
        model: Trained CompactUNet model
        patch_tensor: Input patch tensor (1, 3, 128, 128)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        Tuple of (mean_prediction, uncertainty_map)
    """
    mu_predictions = []
    sigma_sq_predictions = []
    
    # Enable dropout for Monte Carlo sampling
    enable_dropout(model)
    
    with torch.no_grad():
        for _ in range(n_samples):
            mu = model(patch_tensor)
            mu_predictions.append(mu.cpu().numpy())
            # No sigma_sq or transform_params from this model
            sigma_sq_predictions.append(np.zeros_like(mu.cpu().numpy()))
    
    # Disable dropout after sampling
    disable_dropout(model)
    
    # Convert to numpy arrays: (n_samples, 1, 3, 32, 32)
    mu_predictions = np.array(mu_predictions)
    sigma_sq_predictions = np.array(sigma_sq_predictions)
    
    # Calculate mean across samples for mu (epistemic uncertainty)
    mean_mu = np.mean(mu_predictions, axis=0)  # (1, 3, 32, 32)
    var_mu = np.var(mu_predictions, axis=0)    # (1, 3, 32, 32) - epistemic uncertainty
    
    # Average the predicted aleatoric uncertainty (sigma_sq)
    mean_sigma_sq = np.mean(sigma_sq_predictions, axis=0)  # (1, 3, 32, 32)
    
    # Total uncertainty = epistemic + aleatoric
    total_uncertainty = var_mu + mean_sigma_sq
    
    return mean_mu, total_uncertainty


def tile_based_inference(image: np.ndarray, model: torch.nn.Module, device: torch.device, 
                        patch_size: int = 32, window_size: int = 128, 
                        collect_transforms: bool = False) -> tuple:
    """
    Perform tile-based inference on the input image.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model: Trained CompactUNet model
        device: PyTorch device
        patch_size: Size of output patches (32x32)
        window_size: Size of input window (128x128)
    
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
    
    # Collect transform parameters if requested
    transform_data = [] if collect_transforms else None
    
    print(f"Input image shape: {image.shape}")
    print(f"Padded image shape: {padded_image.shape}")
    print(f"Output image shape: {mean_output.shape}")
    print(f"Number of tiles: {num_tiles_y} x {num_tiles_x} = {num_tiles_y * num_tiles_x}")
    print(f"Patch size: {patch_size}x{patch_size}, Window size: {window_size}x{window_size}")
    print(f"Input stride: {input_stride}x{input_stride}")
    
    with torch.no_grad():
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                # Calculate input tile coordinates (stride by window_size)
                input_y_start = tile_y * input_stride
                input_x_start = tile_x * input_stride
                
                # Extract 128x128 input patch
                input_patch = padded_image[input_y_start:input_y_start + window_size,
                                         input_x_start:input_x_start + window_size, :]
                
                # Ensure patch is the right size
                if input_patch.shape[:2] != (window_size, window_size):
                    print(f"Warning: Patch at tile ({tile_y}, {tile_x}) has wrong size: {input_patch.shape}")
                    continue
                
                # Preprocess patch
                patch_tensor = preprocess_patch(input_patch).to(device)
                
                # Get model prediction (32x32 patch only)
                mu = model(patch_tensor)
                
                # Convert predictions back to numpy
                # mu shape: (1, 3, 32, 32)
                patch_mean = mu.squeeze(0).cpu().numpy().transpose(1, 2, 0)      # (32, 32, 3)
                
                # No transform parameters available from this model
                if collect_transforms:
                    # Set default transform values since model doesn't predict them
                    transform_vals = np.array([0.0, 0.0, 0.0, 0.0])  # [x_scale, y_scale, x_offset, y_offset]
                    transform_data.append({
                        'tile_y': tile_y,
                        'tile_x': tile_x,
                        'x_scale': transform_vals[0],
                        'y_scale': transform_vals[1],
                        'x_offset': transform_vals[2],
                        'y_offset': transform_vals[3],
                        'output_y_start': tile_y * patch_size,
                        'output_x_start': tile_x * patch_size
                    })
                
                # Store in output arrays (coordinates in output space)
                output_y_start = tile_y * patch_size
                output_x_start = tile_x * patch_size
                output_y_end = output_y_start + patch_size
                output_x_end = output_x_start + patch_size
                mean_output[output_y_start:output_y_end, output_x_start:output_x_end, :] = patch_mean
                #variance_output[output_y_start:output_y_end, output_x_start:output_x_end, :] = patch_variance
                
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
    
    if collect_transforms:
        return mean_output, variance_output, transform_data
    else:
        return mean_output, variance_output


def tile_based_inference_mc_dropout(image: np.ndarray, model: torch.nn.Module, device: torch.device, 
                                   patch_size: int = 32, window_size: int = 128, 
                                   n_samples: int = 10) -> tuple:
    """
    Perform tile-based inference with Monte Carlo dropout for uncertainty estimation.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model: Trained CompactUNet model
        device: PyTorch device
        patch_size: Size of output patches (32x32)
        window_size: Size of input window (128x128)
        n_samples: Number of Monte Carlo dropout samples
    
    Returns:
        Tuple of (mean_output, uncertainty_output) both at 1/4 resolution of input
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
    uncertainty_output = np.zeros((output_height, output_width, channels), dtype=np.float32)
    
    print(f"Monte Carlo Dropout Inference:")
    print(f"Input image shape: {image.shape}")
    print(f"Padded image shape: {padded_image.shape}")
    print(f"Output image shape: {mean_output.shape}")
    print(f"Number of tiles: {num_tiles_y} x {num_tiles_x} = {num_tiles_y * num_tiles_x}")
    print(f"Patch size: {patch_size}x{patch_size}, Window size: {window_size}x{window_size}")
    print(f"Input stride: {input_stride}x{input_stride}")
    print(f"Monte Carlo samples: {n_samples}")
    
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            # Calculate input tile coordinates (stride by window_size)
            input_y_start = tile_y * input_stride
            input_x_start = tile_x * input_stride
            
            # Extract 128x128 input patch
            input_patch = padded_image[input_y_start:input_y_start + window_size,
                                     input_x_start:input_x_start + window_size, :]
            
            # Ensure patch is the right size
            if input_patch.shape[:2] != (window_size, window_size):
                print(f"Warning: Patch at tile ({tile_y}, {tile_x}) has wrong size: {input_patch.shape}")
                continue
            
            # Preprocess patch
            patch_tensor = preprocess_patch(input_patch).to(device)
            
            # Get Monte Carlo dropout predictions
            mean_pred, var_pred = mc_dropout_inference_single_patch(model, patch_tensor, n_samples)
            
            # Convert predictions back to numpy
            # mean_pred shape: (1, 3, 32, 32), var_pred shape: (1, 3, 32, 32)
            patch_mean = mean_pred.squeeze(0).transpose(1, 2, 0)  # (32, 32, 3)
            patch_var = var_pred.squeeze(0).transpose(1, 2, 0)    # (32, 32, 3)
            
            # Store in output arrays (coordinates in output space)
            output_y_start = tile_y * patch_size
            output_x_start = tile_x * patch_size
            output_y_end = output_y_start + patch_size
            output_x_end = output_x_start + patch_size
            mean_output[output_y_start:output_y_end, output_x_start:output_x_end, :] = patch_mean
            uncertainty_output[output_y_start:output_y_end, output_x_start:output_x_end, :] = patch_var
            
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
    uncertainty_output = uncertainty_output[:orig_output_height, :orig_output_width, :]
    
    return mean_output, uncertainty_output


def batch_tile_inference(image: np.ndarray, model: torch.nn.Module, device: torch.device, 
                        patch_size: int = 32, window_size: int = 128, batch_size: int = 16,
                        collect_transforms: bool = False) -> tuple:
    """
    Perform batched tile-based inference for better GPU utilization.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model: Trained CompactUNet model
        device: PyTorch device
        patch_size: Size of output patches (32x32)
        window_size: Size of input window (128x128)
        batch_size: Number of tiles to process in parallel
    
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
    
    # Collect transform parameters if requested
    transform_data = [] if collect_transforms else None
    
    print(f"Input image shape: {image.shape}")
    print(f"Padded image shape: {padded_image.shape}")
    print(f"Output image shape: {mean_output.shape}")
    print(f"Number of tiles: {num_tiles_y} x {num_tiles_x} = {total_tiles}")
    print(f"Patch size: {patch_size}x{patch_size}, Window size: {window_size}x{window_size}")
    print(f"Input stride: {input_stride}x{input_stride}")
    print(f"Batch size: {batch_size}")
    
    # Collect all tiles and coordinates
    tiles = []
    coordinates = []
    
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            # Calculate input tile coordinates (stride by window_size)
            input_y_start = tile_y * input_stride
            input_x_start = tile_x * input_stride
            
            # Extract 128x128 input patch
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
                preprocess_patch(tile).squeeze(0) for tile in batch_tiles
            ]).to(device)
            
            # Get model predictions
            mu_batch = model(batch_tensor)  # Shape: (batch_size, 3, 32, 32)
            
            # Store results
            for idx, (tile_y, tile_x) in enumerate(batch_coords):
                patch_mean = mu_batch[idx].cpu().numpy().transpose(1, 2, 0)      # (32, 32, 3)
                patch_variance = np.zeros_like(patch_mean)  # No variance from this model
                
                # Collect transform parameters if requested
                if collect_transforms:
                    # Set default transform values since model doesn't predict them
                    transform_vals = np.array([0.0, 0.0, 0.0, 0.0])  # [x_scale, y_scale, x_offset, y_offset]
                    transform_data.append({
                        'tile_y': tile_y,
                        'tile_x': tile_x,
                        'x_scale': transform_vals[0],
                        'y_scale': transform_vals[1],
                        'x_offset': transform_vals[2],
                        'y_offset': transform_vals[3],
                        'output_y_start': tile_y * patch_size,
                        'output_x_start': tile_x * patch_size
                    })
                
                # Store in output arrays (coordinates in output space)
                output_y_start = tile_y * patch_size
                output_x_start = tile_x * patch_size
                output_y_end = output_y_start + patch_size
                output_x_end = output_x_start + patch_size
                mean_output[output_y_start:output_y_end, output_x_start:output_x_end, :] = patch_mean
                variance_output[output_y_start:output_y_end, output_x_start:output_x_end, :] = patch_variance
            
            # Progress indicator
            progress = batch_end / len(tiles) * 100
            print(f"Progress: {progress:.1f}% ({batch_end}/{len(tiles)} tiles)")
    
    # Remove padding to get back to original dimensions (1/4 scale)
    orig_height, orig_width, _ = padding_info['original_shape']
    orig_output_height = orig_height // 4
    orig_output_width = orig_width // 4
    mean_output = mean_output[:orig_output_height, :orig_output_width, :]
    variance_output = variance_output[:orig_output_height, :orig_output_width, :]
    
    if collect_transforms:
        return mean_output, variance_output, transform_data
    else:
        return mean_output, variance_output


def postprocess_output(output: np.ndarray) -> np.ndarray:
    """Convert output tensor to displayable image."""
    # Clip values to [0, 1] range
    output = np.clip(output, 0, 1)
    # Convert to uint8
    output = (output * 255).astype(np.uint8)
    return output


def main():
    parser = argparse.ArgumentParser(description='Patch-based inference for image reconstruction using MediumUNet')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints-medium-unet/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output image path (default: input_reconstructed.png)')
    # Removed --use-gap since MediumUNet doesn't have a GAP variant
    parser.add_argument('--patch-size', type=int, default=32, help='Output patch size')
    parser.add_argument('--window-size', type=int, default=128, help='Input window size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--use-batched', action='store_true', help='Use batched inference (faster)')
    parser.add_argument('--mc-dropout', action='store_true', help='Use Monte Carlo dropout for uncertainty estimation')
    parser.add_argument('--mc-samples', type=int, default=10, help='Number of Monte Carlo dropout samples (default: 10)')
    parser.add_argument('--uncertainty-output', type=str, help='Path to save uncertainty map (default: input_uncertainty.png)')
    parser.add_argument('--variance-output', type=str, help='Path to save variance map from heteroscedastic model (default: input_variance.png)')
    parser.add_argument('--print-transforms', action='store_true', help='Print table of transform parameters for all patches')
    parser.add_argument('--save-transforms', type=str, help='Save transform parameters table to CSV file')
    
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
    
    print(f"Loading model from: {args.model}")
    model = load_model(args.model, device)
    
    # Determine if we need to collect transforms
    collect_transforms = args.print_transforms or args.save_transforms
    
    # Perform inference
    if args.mc_dropout:
        print("Starting Monte Carlo dropout inference...")
        mean_output, uncertainty_output = tile_based_inference_mc_dropout(
            image_array, model, device,
            patch_size=args.patch_size,
            window_size=args.window_size,
            n_samples=args.mc_samples
        )
        
        # Postprocess outputs
        output_image = postprocess_output(mean_output)
        uncertainty_image = postprocess_output(np.sqrt(uncertainty_output))  # Standard deviation
        transform_data = None  # MC dropout doesn't support transform collection yet
    else:
        print("Starting patch-based inference...")
        if args.use_batched:
            if collect_transforms:
                mean_output, variance_output, transform_data = batch_tile_inference(
                    image_array, model, device, 
                    patch_size=args.patch_size, 
                    window_size=args.window_size,
                    batch_size=args.batch_size,
                    collect_transforms=True)
            else:
                mean_output, variance_output = batch_tile_inference(
                    image_array, model, device, 
                    patch_size=args.patch_size, 
                    window_size=args.window_size,
                    batch_size=args.batch_size)
                transform_data = None
        else:
            if collect_transforms:
                mean_output, variance_output, transform_data = tile_based_inference(
                    image_array, model, device, 
                    patch_size=args.patch_size, 
                    window_size=args.window_size,
                    collect_transforms=True)
            else:
                mean_output, variance_output = tile_based_inference(
                    image_array, model, device, 
                    patch_size=args.patch_size, 
                    window_size=args.window_size)
                transform_data = None
        
        # Postprocess outputs
        output_image = postprocess_output(mean_output)
        variance_image = postprocess_output(np.sqrt(variance_output))  # Standard deviation
        uncertainty_image = None
    
    # Save output
    if args.output is None:
        input_path = Path(args.input_image)
        output_path = input_path.parent / f"{input_path.stem}_reconstructed{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    Image.fromarray(output_image).save(output_path)
    print(f"Reconstructed image saved to: {output_path}")
    
    # Save uncertainty map if Monte Carlo dropout was used
    if args.mc_dropout and uncertainty_image is not None:
        if args.uncertainty_output is None:
            input_path = Path(args.input_image)
            uncertainty_path = input_path.parent / f"{input_path.stem}_uncertainty{input_path.suffix}"
        else:
            uncertainty_path = Path(args.uncertainty_output)
        
        Image.fromarray(uncertainty_image).save(uncertainty_path)
        print(f"Uncertainty map saved to: {uncertainty_path}")
    
    # Save variance map from heteroscedastic model
    if not args.mc_dropout and 'variance_image' in locals():
        if args.variance_output is None:
            input_path = Path(args.input_image)
            variance_path = input_path.parent / f"{input_path.stem}_variance{input_path.suffix}"
        else:
            variance_path = Path(args.variance_output)
        
        Image.fromarray(variance_image).save(variance_path)
        print(f"Variance map saved to: {variance_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Input resolution: {image_array.shape[1]}x{image_array.shape[0]}")
    print(f"Output resolution: {output_image.shape[1]}x{output_image.shape[0]}")
    print(f"Model: MediumUNet")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Window size: {args.window_size}x{args.window_size}")
    if args.mc_dropout:
        print(f"Monte Carlo dropout: {args.mc_samples} samples")
        if uncertainty_image is not None:
            uncertainty_stats = np.sqrt(uncertainty_output)
            print(f"Uncertainty statistics (std dev):")
            print(f"  Mean: {np.mean(uncertainty_stats):.4f}")
            print(f"  Min: {np.min(uncertainty_stats):.4f}")
            print(f"  Max: {np.max(uncertainty_stats):.4f}")
    else:
        if 'variance_output' in locals():
            variance_stats = np.sqrt(variance_output)
            print(f"Heteroscedastic variance statistics (std dev):")
            print(f"  Mean: {np.mean(variance_stats):.4f}")
            print(f"  Min: {np.min(variance_stats):.4f}")
            print(f"  Max: {np.max(variance_stats):.4f}")
    
    # Handle transform parameters table
    if transform_data is not None:
        df = pd.DataFrame(transform_data)
        
        if args.print_transforms:
            print(f"\nTransform Parameters Table:")
            print("=" * 80)
            print(df.to_string(index=False, float_format='%.4f'))
            
            # Summary statistics
            print(f"\nTransform Parameter Statistics:")
            print(f"X Scale - Mean: {df['x_scale'].mean():.4f}, Std: {df['x_scale'].std():.4f}, Range: [{df['x_scale'].min():.4f}, {df['x_scale'].max():.4f}]")
            print(f"Y Scale - Mean: {df['y_scale'].mean():.4f}, Std: {df['y_scale'].std():.4f}, Range: [{df['y_scale'].min():.4f}, {df['y_scale'].max():.4f}]")
            print(f"X Offset - Mean: {df['x_offset'].mean():.4f}, Std: {df['x_offset'].std():.4f}, Range: [{df['x_offset'].min():.4f}, {df['x_offset'].max():.4f}]")
            print(f"Y Offset - Mean: {df['y_offset'].mean():.4f}, Std: {df['y_offset'].std():.4f}, Range: [{df['y_offset'].min():.4f}, {df['y_offset'].max():.4f}]")
        
        if args.save_transforms:
            df.to_csv(args.save_transforms, index=False)
            print(f"Transform parameters saved to: {args.save_transforms}")
    elif collect_transforms:
        print("Warning: Transform collection was requested but not supported for the current inference mode.")


if __name__ == "__main__":
    main()
