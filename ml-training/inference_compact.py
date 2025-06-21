#!/usr/bin/env python3
"""
Sliding window inference script for pixel reconstruction using CompactUNet.

Takes a large image and reconstructs it using the trained CompactUNet model.
The model takes 129x129 patches and outputs 1x1 pixels (center pixel prediction).
Sliding window moves 1 pixel at a time for dense reconstruction.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import sys
import os

from compact_unet import CompactUNet, CompactUNetGAP


def load_model(checkpoint_path: str, device: torch.device, use_gap: bool = False):
    """Load the trained CompactUNet model from checkpoint."""
    if use_gap:
        model = CompactUNetGAP()
    else:
        model = CompactUNet()
    
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


def pad_image_for_inference(image: np.ndarray, window_size: int = 129) -> tuple:
    """
    Pad image to ensure all pixels can be reconstructed.
    
    Args:
        image: Input image as numpy array (H, W, C)
        window_size: Size of input window (129x129)
    
    Returns:
        Tuple of (padded_image, padding_info)
    """
    height, width, channels = image.shape
    
    # Calculate padding needed
    pad_size = window_size // 2  # 64 pixels on each side
    
    # Pad the image
    padded_image = np.pad(image, 
                         ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
                         mode='reflect')
    
    padding_info = {
        'pad_size': pad_size,
        'original_shape': (height, width, channels)
    }
    
    return padded_image, padding_info


def sliding_window_inference(image: np.ndarray, model: torch.nn.Module, device: torch.device, 
                           window_size: int = 129, stride: int = 1) -> np.ndarray:
    """
    Perform sliding window inference on the input image.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model: Trained CompactUNet model
        device: PyTorch device
        window_size: Size of input window (129x129)
        stride: Step size for sliding window (1 pixel for dense reconstruction)
    
    Returns:
        Reconstructed image as numpy array
    """
    # Pad image to handle edges
    padded_image, padding_info = pad_image_for_inference(image, window_size)
    height, width, channels = padded_image.shape
    
    # Calculate output dimensions (same as original image)
    orig_height, orig_width, _ = padding_info['original_shape']
    
    # Initialize output array
    output = np.zeros((orig_height, orig_width, channels), dtype=np.float32)
    
    print(f"Input image shape: {image.shape}")
    print(f"Padded image shape: {padded_image.shape}")
    print(f"Output image shape: {output.shape}")
    print(f"Processing {orig_height * orig_width} patches...")
    
    with torch.no_grad():
        for i in range(orig_height):
            for j in range(orig_width):
                # Calculate patch coordinates in padded image
                y_start = i  # No additional offset needed due to padding
                x_start = j
                y_end = y_start + window_size
                x_end = x_start + window_size
                
                # Extract patch
                patch = padded_image[y_start:y_end, x_start:x_end, :]
                
                # Ensure patch is the right size
                if patch.shape[:2] != (window_size, window_size):
                    print(f"Warning: Patch at ({i}, {j}) has wrong size: {patch.shape}")
                    continue
                
                # Preprocess patch
                patch_tensor = preprocess_patch(patch).to(device)
                
                # Get model prediction (center pixel)
                prediction = model(patch_tensor)
                
                # Convert prediction back to numpy
                # prediction shape: (1, 3, 1, 1)
                pixel_value = prediction.squeeze().cpu().numpy()
                
                # Store in output array
                output[i, j, :] = pixel_value
                
                # Progress indicator
                if (i * orig_width + j + 1) % 1000 == 0:
                    progress = (i * orig_width + j + 1) / (orig_height * orig_width) * 100
                    print(f"Progress: {progress:.1f}%")
    
    return output


def batch_sliding_window_inference(image: np.ndarray, model: torch.nn.Module, device: torch.device, 
                                  window_size: int = 129, batch_size: int = 16) -> np.ndarray:
    """
    Perform batched sliding window inference for better GPU utilization.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model: Trained CompactUNet model
        device: PyTorch device
        window_size: Size of input window (129x129)
        batch_size: Number of patches to process in parallel
    
    Returns:
        Reconstructed image as numpy array
    """
    # Pad image to handle edges
    padded_image, padding_info = pad_image_for_inference(image, window_size)
    height, width, channels = padded_image.shape
    
    # Calculate output dimensions (same as original image)
    orig_height, orig_width, _ = padding_info['original_shape']
    
    # Initialize output array
    output = np.zeros((orig_height, orig_width, channels), dtype=np.float32)
    
    print(f"Input image shape: {image.shape}")
    print(f"Padded image shape: {padded_image.shape}")
    print(f"Output image shape: {output.shape}")
    print(f"Processing {orig_height * orig_width} patches in batches of {batch_size}...")
    
    # Collect all patches first
    patches = []
    coordinates = []
    
    for i in range(orig_height):
        for j in range(orig_width):
            # Calculate patch coordinates in padded image
            y_start = i
            x_start = j
            y_end = y_start + window_size
            x_end = x_start + window_size
            
            # Extract patch
            patch = padded_image[y_start:y_end, x_start:x_end, :]
            
            if patch.shape[:2] == (window_size, window_size):
                patches.append(patch)
                coordinates.append((i, j))
    
    # Process patches in batches
    total_patches = len(patches)
    
    with torch.no_grad():
        for batch_start in range(0, total_patches, batch_size):
            batch_end = min(batch_start + batch_size, total_patches)
            batch_patches = patches[batch_start:batch_end]
            batch_coords = coordinates[batch_start:batch_end]
            
            # Preprocess batch
            batch_tensor = torch.stack([
                preprocess_patch(patch).squeeze(0) for patch in batch_patches
            ]).to(device)
            
            # Get model predictions
            predictions = model(batch_tensor)
            
            # Store results
            for idx, (i, j) in enumerate(batch_coords):
                pixel_value = predictions[idx].squeeze().cpu().numpy()
                output[i, j, :] = pixel_value
            
            # Progress indicator
            progress = batch_end / total_patches * 100
            print(f"Progress: {progress:.1f}%")
    
    return output


def postprocess_output(output: np.ndarray) -> np.ndarray:
    """Convert output tensor to displayable image."""
    # Clip values to [0, 1] range
    output = np.clip(output, 0, 1)
    # Convert to uint8
    output = (output * 255).astype(np.uint8)
    return output


def main():
    parser = argparse.ArgumentParser(description='Sliding window inference for pixel reconstruction using CompactUNet')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints-compact-unet/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output image path (default: input_reconstructed.png)')
    parser.add_argument('--use-gap', action='store_true', help='Use GAP variant of the model')
    parser.add_argument('--window-size', type=int, default=129, help='Input window size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--use-batched', action='store_true', help='Use batched inference (faster)')
    
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
    model = load_model(args.model, device, use_gap=args.use_gap)
    
    # Perform inference
    print("Starting sliding window inference...")
    if args.use_batched:
        output = batch_sliding_window_inference(image_array, model, device, 
                                              window_size=args.window_size, 
                                              batch_size=args.batch_size)
    else:
        output = sliding_window_inference(image_array, model, device, 
                                        window_size=args.window_size, stride=1)
    
    # Postprocess output
    output_image = postprocess_output(output)
    
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
    print(f"Model: {'CompactUNetGAP' if args.use_gap else 'CompactUNet'}")
    print(f"Window size: {args.window_size}x{args.window_size}")


if __name__ == "__main__":
    main()