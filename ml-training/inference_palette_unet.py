#!/usr/bin/env python3
"""
Inference script for PaletteUNet model.

This script loads a trained PaletteUNet model and performs inference on input images,
reconstructing them using palette-based classification.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our modules
from palette_unet import PaletteUNet, logits_to_palette_indices, palette_indices_to_rgb
from palette_utils import generate_standard_64_palette, indices_to_rgb
from data_synthesis_pipeline import PixelArtDataSynthesizer


def load_model_and_palette(checkpoint_path: str, device: torch.device) -> Tuple[PaletteUNet, np.ndarray, torch.Tensor]:
    """
    Load trained model and palette from checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        device: Device to load model on
    
    Returns:
        Tuple of (model, palette_np, palette_tensor)
    """
    # If checkpoint_path is a directory, look for best_model.pth
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, 'best_model.pth')
        palette_file = os.path.join(checkpoint_path, 'palette.npy')
    else:
        # checkpoint_path is a file
        checkpoint_file = checkpoint_path
        checkpoint_dir = os.path.dirname(checkpoint_path)
        palette_file = os.path.join(checkpoint_dir, 'palette.npy')
    
    # Load palette
    if os.path.exists(palette_file):
        palette_np = np.load(palette_file)
        print(f"Loaded palette from {palette_file}")
    else:
        print(f"Palette file not found at {palette_file}, generating standard 64-color palette")
        palette_np = generate_standard_64_palette()
    
    palette_tensor = torch.from_numpy(palette_np).float() / 255.0  # Convert to [0, 1] range
    palette_tensor = palette_tensor.to(device)
    
    # Load model
    model = PaletteUNet(num_palette_colors=len(palette_np))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_file}")
    print(f"Model trained for {checkpoint['epoch']} epochs with loss {checkpoint['loss']:.6f}")
    
    return model, palette_np, palette_tensor


def preprocess_image(image: Image.Image, target_size: int = 128) -> torch.Tensor:
    """
    Preprocess input image for model inference.
    
    Args:
        image: Input PIL Image
        target_size: Target size for model input (default 128)
    
    Returns:
        Preprocessed tensor (1, 3, target_size, target_size)
    """
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to target size
    image = image.resize((target_size, target_size), Image.LANCZOS)
    
    # Convert to tensor and normalize to [0, 1]
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def postprocess_indices(indices: torch.Tensor, palette_np: np.ndarray) -> Image.Image:
    """
    Convert model output indices to RGB image.
    
    Args:
        indices: Palette indices tensor (H, W)
        palette_np: Palette array (num_colors, 3) with values [0, 255]
    
    Returns:
        RGB PIL Image
    """
    # Convert to numpy
    indices_np = indices.cpu().numpy()
    
    # Convert indices to RGB
    rgb_array = indices_to_rgb(indices_np, palette_np)
    
    # Convert to PIL Image
    return Image.fromarray(rgb_array.astype(np.uint8))


def extract_center_region(image: Image.Image, center_size: int = 32) -> Image.Image:
    """
    Extract center region from image.
    
    Args:
        image: Input image
        center_size: Size of center region to extract
    
    Returns:
        Center region as PIL Image
    """
    width, height = image.size
    center_x, center_y = width // 2, height // 2
    half_size = center_size // 2
    
    left = center_x - half_size
    top = center_y - half_size
    right = center_x + half_size
    bottom = center_y + half_size
    
    return image.crop((left, top, right, bottom))


def run_inference(model: PaletteUNet, 
                 input_tensor: torch.Tensor, 
                 palette_tensor: torch.Tensor,
                 palette_np: np.ndarray) -> Tuple[torch.Tensor, Image.Image]:
    """
    Run inference on input tensor.
    
    Args:
        model: Trained PaletteUNet model
        input_tensor: Input tensor (1, 3, H, W)
        palette_tensor: Palette tensor (num_colors, 3)
        palette_np: Palette array for postprocessing
    
    Returns:
        Tuple of (predicted_indices, reconstructed_image)
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # Run model inference
        logits = model(input_tensor)
        
        # Convert logits to indices
        predicted_indices = logits_to_palette_indices(logits)
        
        # Remove batch dimension
        predicted_indices = predicted_indices.squeeze(0)
        
        # Convert to RGB image
        reconstructed_image = postprocess_indices(predicted_indices, palette_np)
    
    return predicted_indices, reconstructed_image


def create_comparison_plot(input_image: Image.Image, 
                          reconstructed_image: Image.Image,
                          save_path: Optional[str] = None) -> None:
    """
    Create a comparison plot showing input vs reconstruction.
    
    Args:
        input_image: Original input image
        reconstructed_image: Reconstructed image
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image (full size)
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image (128x128)')
    axes[0].axis('off')
    
    # Input center region for comparison
    input_center = extract_center_region(input_image, 32)
    axes[1].imshow(input_center)
    axes[1].set_title('Input Center (32x32)')
    axes[1].axis('off')
    
    # Reconstructed image
    axes[2].imshow(reconstructed_image)
    axes[2].set_title('Reconstructed (32x32)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained PaletteUNet')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file or directory')
    parser.add_argument('--input_image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results')
    
    # Processing arguments
    parser.add_argument('--input_size', type=int, default=128,
                       help='Input size for model (default: 128)')
    parser.add_argument('--show_plot', action='store_true',
                       help='Show comparison plot instead of saving')
    
    # Device arguments
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='auto',
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and palette
    model, palette_np, palette_tensor = load_model_and_palette(args.checkpoint, device)
    
    # Load and preprocess input image
    input_image = Image.open(args.input_image)
    input_tensor = preprocess_image(input_image, args.input_size)
    
    print(f"Input image: {args.input_image}")
    print(f"Input size: {input_image.size} -> {input_tensor.shape}")
    
    # Run inference
    predicted_indices, reconstructed_image = run_inference(
        model, input_tensor, palette_tensor, palette_np
    )
    
    print(f"Reconstruction size: {reconstructed_image.size}")
    print(f"Unique palette indices used: {len(torch.unique(predicted_indices))}")
    
    # Save results
    base_name = Path(args.input_image).stem
    
    # Save reconstructed image
    reconstructed_path = os.path.join(args.output_dir, f"{base_name}_reconstructed.png")
    reconstructed_image.save(reconstructed_path)
    print(f"Reconstructed image saved to {reconstructed_path}")
    
    # Save input image for reference
    input_path = os.path.join(args.output_dir, f"{base_name}_input.png")
    input_image.resize((args.input_size, args.input_size)).save(input_path)
    
    # Save center region for comparison
    input_center = extract_center_region(input_image.resize((args.input_size, args.input_size)), 32)
    center_path = os.path.join(args.output_dir, f"{base_name}_center.png")
    input_center.save(center_path)
    
    # Create and save comparison plot
    if not args.show_plot:
        comparison_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
        create_comparison_plot(
            input_image.resize((args.input_size, args.input_size)), 
            reconstructed_image, 
            comparison_path
        )
    else:
        create_comparison_plot(
            input_image.resize((args.input_size, args.input_size)), 
            reconstructed_image
        )
    
    # Save palette indices as numpy array
    indices_path = os.path.join(args.output_dir, f"{base_name}_indices.npy")
    np.save(indices_path, predicted_indices.cpu().numpy())
    print(f"Palette indices saved to {indices_path}")
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()