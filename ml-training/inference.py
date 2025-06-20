#!/usr/bin/env python3
"""
Sliding window inference script for pixel reconstruction.

Takes a large image and reconstructs it using the trained PixelUNet model.
The model takes 128x128 patches and outputs 1x1 pixels.
Since output is 1/4 resolution, input window moves 4 pixels per output pixel.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import sys
import os

from pixel_unet import PixelUNet, PixelUNetGAP


def load_model(checkpoint_path: str, device: torch.device, use_gap: bool = False):
    """Load the trained model from checkpoint."""
    if use_gap:
        model = PixelUNetGAP()
    else:
        model = PixelUNet()
    
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


def sliding_window_inference(image: np.ndarray, model: torch.nn.Module, device: torch.device, 
                           window_size: int = 128, stride: int = 4) -> np.ndarray:
    """
    Perform sliding window inference on the input image.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model: Trained PixelUNet model
        device: PyTorch device
        window_size: Size of input window (128x128)
        stride: Step size for sliding window (4 pixels)
    
    Returns:
        Reconstructed image as numpy array
    """
    height, width, channels = image.shape
    
    # Calculate output dimensions
    output_height = (height - window_size) // stride + 1
    output_width = (width - window_size) // stride + 1
    
    # Initialize output array
    output = np.zeros((output_height, output_width, channels), dtype=np.float32)
    
    print(f"Input image shape: {image.shape}")
    print(f"Output image shape: {output.shape}")
    print(f"Processing {output_height * output_width} patches...")
    
    with torch.no_grad():
        for i in range(output_height):
            for j in range(output_width):
                # Calculate patch coordinates
                y_start = i * stride
                x_start = j * stride
                y_end = y_start + window_size
                x_end = x_start + window_size
                
                # Extract patch
                patch = image[y_start:y_end, x_start:x_end, :]
                
                # Skip if patch is not the right size (edge cases)
                if patch.shape[:2] != (window_size, window_size):
                    continue
                
                # Preprocess patch
                patch_tensor = preprocess_patch(patch).to(device)
                
                # Get model prediction
                prediction = model(patch_tensor)
                
                # Convert prediction back to numpy
                # prediction shape: (1, 3, 1, 1)
                pixel_value = prediction.squeeze().cpu().numpy()
                
                # Store in output array
                output[i, j, :] = pixel_value
                
                # Progress indicator
                if (i * output_width + j + 1) % 100 == 0:
                    progress = (i * output_width + j + 1) / (output_height * output_width) * 100
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
    parser = argparse.ArgumentParser(description='Sliding window inference for pixel reconstruction')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='ml-training/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output image path (default: input_reconstructed.png)')
    parser.add_argument('--use-gap', action='store_true', help='Use GAP variant of the model')
    parser.add_argument('--window-size', type=int, default=128, help='Input window size')
    parser.add_argument('--stride', type=int, default=4, help='Stride for sliding window')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, auto)')
    
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
    output = sliding_window_inference(image_array, model, device, 
                                    window_size=args.window_size, stride=args.stride)
    
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
    print(f"Compression ratio: {(image_array.shape[0] * image_array.shape[1]) / (output_image.shape[0] * output_image.shape[1]):.1f}x")


if __name__ == "__main__":
    main()
