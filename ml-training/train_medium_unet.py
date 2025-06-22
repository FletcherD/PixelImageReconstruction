#!/usr/bin/env python3
"""
Training script for MediumUNet model using the data synthesis pipeline.
Uses both local images and HuggingFace 'nerijs/pixelparti-128-v0.1' dataset.
Configured for 128x128 input patches with 32x32 center patch prediction.

MediumUNet is a more sophisticated model with ~6.6M parameters vs CompactUNet's ~1.35M.
"""

import os
import argparse
import random
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from tqdm import tqdm
import wandb
import math

# Import our modules
from medium_unet import MediumUNet, count_parameters, mse_loss
from data_synthesis_pipeline import (
    PixelArtDataSynthesizer, 
    PixelArtDataset, 
    HuggingFacePixelArtDataset
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')




def create_datasets(args) -> tuple:
    """Create training and validation datasets."""
    synthesizer = PixelArtDataSynthesizer(
        crop_size=32, 
        input_size=128, 
        target_size=32, 
        seed=args.seed
    )
    
    datasets = []
    
    # Add HuggingFace dataset
    if args.use_hf_dataset:
        print(f"Loading HuggingFace dataset: {args.hf_dataset_name}")
        hf_dataset = HuggingFacePixelArtDataset(
            dataset_name=args.hf_dataset_name,
            num_samples=args.hf_samples,
            synthesizer=synthesizer,
            split='train',
            streaming=args.hf_streaming,
            crop_size=32,
            input_size=128,
            target_size=32
        )
        datasets.append(hf_dataset)
    
    # Add local dataset if specified
    if args.source_images_dir:
        print(f"Loading local images from: {args.source_images_dir}")
        local_dataset = PixelArtDataset(
            source_images_dir=args.source_images_dir,
            num_samples=args.local_samples,
            synthesizer=synthesizer,
            crop_size=32,
            input_size=128,
            target_size=32
        )
        datasets.append(local_dataset)
    
    if not datasets:
        raise ValueError("No datasets specified. Use --use_hf_dataset or --source_images_dir")
    
    # Combine datasets
    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def apply_random_transform(image, transform_params):
    """Apply spatial transformation to input image based on parameters.
    
    """
    x_scale, y_scale, x_offset, y_offset = transform_params
    
    # Create affine transformation matrix
    theta = torch.zeros(1, 2, 3, device=image.device, dtype=image.dtype)
    theta[0, 0, 0] = torch.exp(x_scale * 0.4 - 0.2)
    theta[0, 1, 1] = torch.exp(y_scale * 0.4 - 0.2)
    theta[0, 0, 2] = x_offset / 32.0
    theta[0, 1, 2] = y_offset / 32.0
    
    # Apply transformation
    grid = torch.nn.functional.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
    transformed = torch.nn.functional.grid_sample(image.unsqueeze(0), grid, align_corners=False)
    
    return transformed.squeeze(0)


def generate_transform_params(batch_size, device, transform_fraction=0.5):
    """Generate random transformation parameters for training."""
    # Control fraction of transformed samples
    transformed_samples = int(batch_size * transform_fraction)
    aligned_samples = batch_size - transformed_samples
    
    # Aligned samples (all zeros)
    aligned_params = torch.zeros(aligned_samples, 4, device=device)
    
    # Transformed samples
    if transformed_samples > 0:
        # Random shift in circular pattern (-1 to 1)
        angles = torch.rand(transformed_samples, device=device) * 2 * math.pi
        distances = torch.rand(transformed_samples, device=device)
        x_shift_pixels = distances * torch.cos(angles)
        y_shift_pixels = distances * torch.sin(angles)
        
        x_offset = x_shift_pixels
        y_offset = y_shift_pixels
        
        # Random scale
        x_scale = torch.rand(transformed_samples, device=device) * 2 - 1
        y_scale = torch.rand(transformed_samples, device=device) * 2 - 1
        
        transformed_params = torch.stack([x_scale, y_scale, x_offset, y_offset], dim=1)
    else:
        transformed_params = torch.empty(0, 4, device=device)
    
    # Combine and shuffle
    all_params = torch.cat([aligned_params, transformed_params], dim=0)
    perm = torch.randperm(batch_size, device=device)
    return all_params[perm]


def transform_loss(pred_transform, true_transform):
    """Compute loss for transformation parameters."""
    return torch.nn.functional.mse_loss(pred_transform, true_transform)


def test_transformation_consistency():
    """Test function to verify transformation coordinate system is correct.
    
    Expected behavior:
    - Shifting input 2 pixels LEFT should give x_offset = +1.0 (in [-1,1] range)
    - Shifting input 2 pixels RIGHT should give x_offset = -1.0 (in [-1,1] range)  
    - Shifting input 2 pixels UP should give y_offset = +1.0 (in [-1,1] range)
    - Shifting input 2 pixels DOWN should give y_offset = -1.0 (in [-1,1] range)
    """
    print("\\nTransformation Coordinate System:")
    print("For 2-pixel shifts on 128x128 input (max range):")
    print("- Left shift  (x_shift = -2): x_offset should be +1.0")
    print("- Right shift (x_shift = +2): x_offset should be -1.0") 
    print("- Up shift    (y_shift = -2): y_offset should be +1.0")
    print("- Down shift  (y_shift = +2): y_offset should be -1.0")
    print("\\nActual calculation:")
    
    # Test cases
    test_cases = [
        ("Left shift", -2, 0),
        ("Right shift", 2, 0), 
        ("Up shift", 0, -2),
        ("Down shift", 0, 2)
    ]
    
    for name, x_shift, y_shift in test_cases:
        x_offset = -x_shift / 64  # Our current formula
        y_offset = -y_shift / 64
        print(f"- {name:11} (x_shift={x_shift:2}, y_shift={y_shift:2}): x_offset={x_offset:+.3f}, y_offset={y_offset:+.3f}")
    print()


def create_validation_image_grid(inputs, targets, transformed_inputs, pred_images, pred_transforms, true_transforms, max_samples=16):
    """Create a grid of validation images for wandb logging.
    
    Args:
        inputs: Original input images (B, C, H, W)
        targets: Target center patches (B, C, H, W) 
        transformed_inputs: Transformed input images (B, C, H, W)
        pred_images: Predicted center patches (B, C, H, W)
        pred_transforms: Predicted transform parameters (B, 4)
        true_transforms: Ground truth transform parameters (B, 4)
        max_samples: Maximum number of samples to include
        
    Returns:
        wandb.Image objects for logging
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    batch_size = min(inputs.size(0), max_samples)
    
    # Create a grid showing: original input center | transformed input center | target | prediction
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Extract center regions for visualization (32x32 from 128x128)
        input_center = inputs[i, :, 48:80, 48:80].cpu().permute(1, 2, 0).numpy()
        transformed_center = transformed_inputs[i, :, 48:80, 48:80].cpu().permute(1, 2, 0).numpy()
        target_img = targets[i].cpu().permute(1, 2, 0).numpy()
        pred_img = pred_images[i].cpu().permute(1, 2, 0).numpy()
        
        # Clip values to [0, 1] for display
        input_center = np.clip(input_center, 0, 1)
        transformed_center = np.clip(transformed_center, 0, 1)
        target_img = np.clip(target_img, 0, 1)
        pred_img = np.clip(pred_img, 0, 1)
        
        # Extract transform parameters
        true_t = true_transforms[i].cpu().numpy()
        pred_t = pred_transforms[i].cpu().numpy()
        
        # Original input center
        axes[i, 0].imshow(input_center)
        axes[i, 0].set_title(f"Original Center\nSample {i}")
        axes[i, 0].axis('off')
        
        # Transformed input center
        axes[i, 1].imshow(transformed_center)
        transform_text = f"True: [{true_t[0]:.2f}, {true_t[1]:.2f}, {true_t[2]:.2f}, {true_t[3]:.2f}]"
        axes[i, 1].set_title(f"Transformed Input\n{transform_text}")
        axes[i, 1].axis('off')
        
        # Target (ground truth)
        axes[i, 2].imshow(target_img)
        axes[i, 2].set_title(f"Target\n(Ground Truth)")
        axes[i, 2].axis('off')
        
        # Prediction
        axes[i, 3].imshow(pred_img)
        pred_text = f"Pred: [{pred_t[0]:.2f}, {pred_t[1]:.2f}, {pred_t[2]:.2f}, {pred_t[3]:.2f}]"
        error = np.abs(pred_t - true_t).mean()
        axes[i, 3].set_title(f"Prediction\n{pred_text}\nError: {error:.3f}")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Convert to wandb Image
    wandb_image = wandb.Image(fig)
    plt.close(fig)
    
    return wandb_image


def test_transform_training_samples(model, device, output_dir="debug_samples"):
    """Test function to output example training samples for debugging transform convergence.
    
    Creates training samples with specific transforms to debug why the 4 scalar outputs
    are failing to converge during training.
    
    Args:
        model: The MediumUNet model to test
        device: Device to run inference on
        output_dir: Directory to save debug samples
    """
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple test image (checkerboard pattern)
    test_image = torch.zeros(3, 128, 128, device=device)
    # Checkerboard pattern
    for i in range(0, 128, 8):
        for j in range(0, 128, 8):
            if (i // 8 + j // 8) % 2 == 0:
                test_image[:, i:i+8, j:j+8] = 1.0
    
    # Define test transforms
    test_transforms = [
        ("zero_transform", [0.0, 0.0, 0.0, 0.0]),  # No transform
        ("x_offset_neg1", [0.0, 0.0, -1.0, 0.0]),  # x offset -1 (shift right 2 pixels)
        ("x_offset_pos1", [0.0, 0.0, 1.0, 0.0]),   # x offset +1 (shift left 2 pixels)
        ("y_offset_neg1", [0.0, 0.0, 0.0, -1.0]),  # y offset -1 (shift down 2 pixels)
        ("y_offset_pos1", [0.0, 0.0, 0.0, 1.0]),   # y offset +1 (shift up 2 pixels)
        ("scale_09", [-0.2, -0.2, 0.0, 0.0]),        # Scale 0.9
        ("scale_11", [0.2, 0.2, 0.0, 0.0]),        # Scale 1.1
    ]
    
    model.eval()
    with torch.no_grad():
        for i, (name, transform_params) in enumerate(test_transforms):
            print(f"\\nTesting {name}: {transform_params}")
            
            # Convert to tensor
            transform_tensor = torch.tensor(transform_params, device=device)
            
            # Apply transform to input
            if torch.allclose(transform_tensor, torch.zeros(4, device=device)):
                transformed_input = test_image
            else:
                transformed_input = apply_random_transform(test_image, transform_tensor)
            
            # Get model prediction
            pred_image, pred_transform = model(transformed_input.unsqueeze(0))
            pred_image = pred_image.squeeze(0)
            pred_transform = pred_transform.squeeze(0)
            
            print(f"  Ground truth transform: {transform_params}")
            print(f"  Predicted transform:    {pred_transform.cpu().numpy()}")
            print(f"  Transform error:        {torch.abs(pred_transform - transform_tensor).cpu().numpy()}")
            
            # Save visualizations
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original center crop (32x32 from center of 128x128)
            original_center = test_image[:, 48:80, 48:80]  # Center 32x32
            axes[0].imshow(original_center.permute(1, 2, 0).cpu().numpy())
            axes[0].set_title(f"Original Center\\n(Ground Truth)")
            axes[0].axis('off')
            
            # Transformed input (show center region)
            transformed_center = transformed_input[:, 48:80, 48:80]
            axes[1].imshow(transformed_center.permute(1, 2, 0).cpu().numpy())
            axes[1].set_title(f"Transformed Input Center\\n{name}")
            axes[1].axis('off')
            
            # Model prediction
            axes[2].imshow(pred_image.permute(1, 2, 0).cpu().numpy())
            axes[2].set_title(f"Model Prediction\\nTransform: {pred_transform.cpu().numpy()}")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{i:02d}_{name}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also save individual images
            def save_tensor_as_image(tensor, filepath):
                # Convert CHW to HWC and ensure values are in [0,1]
                img_array = tensor.permute(1, 2, 0).cpu().numpy()
                img_array = np.clip(img_array, 0, 1)
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                img_pil.save(filepath)
            
            save_tensor_as_image(original_center, os.path.join(output_dir, f"{i:02d}_{name}_original.png"))
            save_tensor_as_image(transformed_center, os.path.join(output_dir, f"{i:02d}_{name}_input.png"))
            save_tensor_as_image(pred_image, os.path.join(output_dir, f"{i:02d}_{name}_pred.png"))
    
    print(f"\\nDebug samples saved to {output_dir}/")
    print("Check the images to verify:")
    print("1. Transform applications are working correctly")
    print("2. Model predictions are reasonable")
    print("3. Transform parameter predictions match ground truth")


def train_epoch(model, dataloader, optimizer, device, epoch: int, 
                transform_fraction: float = 0.5, transform_loss_weight: float = 0.1,
                training_strategy: str = 'joint', transform_only_epoch: bool = False) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_img_loss = 0.0
    total_transform_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Generate random transformation parameters
        true_transform_params = generate_transform_params(batch_size, device, transform_fraction)
        
        # Apply transformations to inputs
        transformed_inputs = []
        for i in range(batch_size):
            if torch.allclose(true_transform_params[i], torch.zeros(4, device=device)):
                # No transformation
                transformed_inputs.append(inputs[i])
            else:
                # Apply transformation
                transformed_inputs.append(apply_random_transform(inputs[i], true_transform_params[i]))
        
        transformed_inputs = torch.stack(transformed_inputs)
        
        optimizer.zero_grad()
        mu, pred_transform_params = model(transformed_inputs)
        
        # Determine which samples to train on for each task
        if training_strategy == 'conditional':
            # Only train image reconstruction on aligned samples, transforms on transformed samples
            aligned_mask = torch.all(true_transform_params == 0, dim=1)
            transformed_mask = ~aligned_mask
        else:
            # Train on all samples for both tasks
            aligned_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            transformed_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Image reconstruction loss (only on aligned samples for conditional training)
        if transform_only_epoch:
            # Skip image loss entirely for transform-only epochs
            img_loss = torch.tensor(0.0, device=device)
        elif training_strategy == 'conditional' and aligned_mask.any():
            img_loss = mse_loss(mu[aligned_mask], targets[aligned_mask])
        elif training_strategy != 'conditional':
            img_loss = mse_loss(mu, targets)
        else:
            img_loss = torch.tensor(0.0, device=device)
        
        # Transform parameter loss (only on transformed samples for conditional training)
        if training_strategy == 'conditional' and transformed_mask.any():
            trans_loss = transform_loss(pred_transform_params[transformed_mask], true_transform_params[transformed_mask])
        elif training_strategy != 'conditional':
            trans_loss = transform_loss(pred_transform_params, true_transform_params)
        else:
            trans_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        if transform_only_epoch:
            loss = trans_loss
        else:
            loss = img_loss + transform_loss_weight * trans_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_img_loss += img_loss.item()
        total_transform_loss += trans_loss.item()
        
        avg_loss = total_loss / (batch_idx + 1)
        avg_img_loss = total_img_loss / (batch_idx + 1)
        avg_trans_loss = total_transform_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.6f}',
            'img': f'{avg_img_loss:.6f}',
            'trans': f'{avg_trans_loss:.6f}'
        })
        
        # Log batch metrics to wandb
        if wandb.run is not None:
            wandb.log({
                'batch_loss': loss.item(),
                'batch_img_loss': img_loss.item(),
                'batch_transform_loss': trans_loss.item(),
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return {
        'train_loss': total_loss / num_batches,
        'train_img_loss': total_img_loss / num_batches,
        'train_transform_loss': total_transform_loss / num_batches
    }


def validate_epoch(model, dataloader, device, epoch: int,
                   transform_fraction: float = 0.5, transform_loss_weight: float = 0.1,
                   training_strategy: str = 'joint', log_images: bool = False, 
                   max_image_samples: int = 16) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_img_loss = 0.0
    total_transform_loss = 0.0
    total_mae = 0.0
    total_uncertainty = 0.0
    total_transform_mae = 0.0
    num_batches = len(dataloader)
    
    # For image logging
    logged_samples = 0
    sample_inputs = []
    sample_targets = []
    sample_transformed_inputs = []
    sample_pred_images = []
    sample_pred_transforms = []
    sample_true_transforms = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f'Val {epoch}')):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # Generate random transformation parameters
            true_transform_params = generate_transform_params(batch_size, device, transform_fraction)
            
            # Apply transformations to inputs
            transformed_inputs = []
            for i in range(batch_size):
                if torch.allclose(true_transform_params[i], torch.zeros(4, device=device)):
                    # No transformation
                    transformed_inputs.append(inputs[i])
                else:
                    # Apply transformation
                    transformed_inputs.append(apply_random_transform(inputs[i], true_transform_params[i]))
            
            transformed_inputs = torch.stack(transformed_inputs)
            
            mu, pred_transform_params = model(transformed_inputs)
            
            # Image losses
            img_loss = mse_loss(mu, targets)
            mae = torch.abs(mu - targets).mean()
            
            # Transform losses
            trans_loss = transform_loss(pred_transform_params, true_transform_params)
            transform_mae = torch.abs(pred_transform_params - true_transform_params).mean()
            
            # Combined loss (always compute full loss during validation)
            loss = img_loss + transform_loss_weight * trans_loss
            
            total_loss += loss.item()
            total_img_loss += img_loss.item()
            total_transform_loss += trans_loss.item()
            total_mae += mae.item()
            total_transform_mae += transform_mae.item()
            
            # Collect samples for image logging
            if log_images and logged_samples < max_image_samples and wandb.run is not None:
                samples_to_take = min(batch_size, max_image_samples - logged_samples)
                sample_inputs.append(inputs[:samples_to_take].cpu())
                sample_targets.append(targets[:samples_to_take].cpu())
                sample_transformed_inputs.append(transformed_inputs[:samples_to_take].cpu())
                sample_pred_images.append(mu[:samples_to_take].cpu())
                sample_pred_transforms.append(pred_transform_params[:samples_to_take].cpu())
                sample_true_transforms.append(true_transform_params[:samples_to_take].cpu())
                logged_samples += samples_to_take
    
    # Log validation images to wandb
    if log_images and logged_samples > 0 and wandb.run is not None:
        # Concatenate all collected samples
        all_inputs = torch.cat(sample_inputs, dim=0)
        all_targets = torch.cat(sample_targets, dim=0)
        all_transformed_inputs = torch.cat(sample_transformed_inputs, dim=0)
        all_pred_images = torch.cat(sample_pred_images, dim=0)
        all_pred_transforms = torch.cat(sample_pred_transforms, dim=0)
        all_true_transforms = torch.cat(sample_true_transforms, dim=0)
        
        # Create validation image grid
        validation_image = create_validation_image_grid(
            all_inputs, all_targets, all_transformed_inputs,
            all_pred_images, all_pred_transforms, all_true_transforms,
            max_samples=logged_samples
        )
        
        # Log to wandb
        wandb.log({
            "validation_samples": validation_image,
            "epoch": epoch
        })
    
    return {
        'val_loss': total_loss / num_batches,
        'val_img_loss': total_img_loss / num_batches,
        'val_transform_loss': total_transform_loss / num_batches,
        'val_mae': total_mae / num_batches,
        'val_transform_mae': total_transform_mae / num_batches
    }


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def main():
    parser = argparse.ArgumentParser(description='Train MediumUNet for patch prediction with transform detection')
    
    # Dataset arguments
    parser.add_argument('--source_images_dir', type=str, 
                       help='Directory containing local source images')
    parser.add_argument('--local_samples', type=int, default=10000,
                       help='Number of samples from local dataset')
    parser.add_argument('--use_hf_dataset', action='store_true',
                       help='Use HuggingFace dataset')
    parser.add_argument('--hf_dataset_name', type=str, default='nerijs/pixelparti-128-v0.1',
                       help='HuggingFace dataset name')
    parser.add_argument('--hf_samples', type=int, default=10000,
                       help='Number of samples from HF dataset')
    parser.add_argument('--hf_streaming', action='store_true',
                       help='Use streaming mode for HF dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (increased for compact model)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--transform_fraction', type=float, default=0.5,
                       help='Fraction of samples with applied transformations (0.0-1.0)')
    parser.add_argument('--transform_loss_weight', type=float, default=0.1,
                       help='Weight for transform parameter loss relative to image loss')
    parser.add_argument('--training_strategy', type=str, default='joint',
                       choices=['joint', 'conditional', 'alternating'],
                       help='Training strategy: joint (both tasks), conditional (image on aligned, transform on transformed), alternating (separate phases)')
    parser.add_argument('--alternating_epochs', type=int, default=5,
                       help='Number of epochs per phase in alternating strategy')
    parser.add_argument('--transform_warmup_epochs', type=int, default=0,
                       help='Number of epochs to train only transforms before joint training')
    
    # Model arguments (MediumUNet only has one type)
    # Removed model_type since MediumUNet doesn't have a GAP variant
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--use_transpose', action='store_true', default=True,
                       help='Use transposed convolution for upsampling')
    parser.add_argument('--final_activation', type=str, default='sigmoid',
                       choices=['sigmoid', 'tanh', 'none'],
                       help='Final activation function')
    
    
    # Training setup
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints-medium-unet',
                       help='Output directory for checkpoints')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='medium-unet',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable wandb logging')
    
    # Debug arguments
    parser.add_argument('--test_transforms', action='store_true',
                       help='Run transform debugging test and exit')
    parser.add_argument('--log_validation_images', action='store_true',
                       help='Log validation images to wandb')
    parser.add_argument('--max_validation_images', type=int, default=16,
                       help='Maximum number of validation images to log per epoch')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize wandb
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = MediumUNet(
        dropout=args.dropout,
        use_transpose=args.use_transpose,
        final_activation=args.final_activation
    )
    
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Run transform debugging test if requested
    if args.test_transforms:
        print("Running transform debugging test...")
        test_transform_training_samples(model, device, "debug_transform_samples")
        print("Transform debugging test completed. Exiting.")
        return
    
    # Create optimizer (using heteroscedastic_loss function directly)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Log model architecture to wandb
    if not args.disable_wandb:
        wandb.watch(model, log_freq=100)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Determine training strategy for this epoch
        if args.training_strategy == 'alternating':
            # Alternating strategy: switch between image-only and transform-only phases
            phase_epoch = ((epoch - 1) // args.alternating_epochs) % 2
            if phase_epoch == 0:
                # Image reconstruction phase - train only on aligned samples
                current_strategy = 'conditional'
                current_transform_fraction = 0.0  # Only aligned samples
                transform_only_epoch = False
                print(f"Epoch {epoch}: Image reconstruction phase")
            else:
                # Transform detection phase - train only transforms
                current_strategy = 'conditional'
                current_transform_fraction = 1.0  # Only transformed samples
                transform_only_epoch = True
                print(f"Epoch {epoch}: Transform detection phase")
        elif epoch <= args.transform_warmup_epochs:
            # Transform warmup phase
            current_strategy = 'conditional'
            current_transform_fraction = 1.0
            transform_only_epoch = True
            print(f"Epoch {epoch}: Transform warmup phase")
        else:
            # Normal training
            current_strategy = args.training_strategy
            current_transform_fraction = args.transform_fraction
            transform_only_epoch = False
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch,
                                  transform_fraction=current_transform_fraction,
                                  transform_loss_weight=args.transform_loss_weight,
                                  training_strategy=current_strategy,
                                  transform_only_epoch=transform_only_epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, epoch,
                                   transform_fraction=args.transform_fraction,
                                   transform_loss_weight=args.transform_loss_weight,
                                   training_strategy=args.training_strategy,
                                   log_images=args.log_validation_images and not args.disable_wandb,
                                   max_image_samples=args.max_validation_images)
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        
        print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.6f}, "
              f"Val Loss: {val_metrics['val_loss']:.6f}, Val MAE: {val_metrics['val_mae']:.6f}, "
              f"Transform MAE: {val_metrics['val_transform_mae']:.6f}")
        
        if not args.disable_wandb:
            wandb.log(metrics)
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics['val_loss'],
                os.path.join(args.output_dir, 'best_model.pth')
            )
        
        # Save periodic checkpoints
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['val_loss'],
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs, val_metrics['val_loss'],
        os.path.join(args.output_dir, 'final_model.pth')
    )
    
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    
    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
