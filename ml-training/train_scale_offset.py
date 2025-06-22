#!/usr/bin/env python3
"""
Training script for Scale/Offset detection model using the data synthesis pipeline.
Uses both local images and HuggingFace 'nerijs/pixelparti-128-v0.1' dataset.
Configured for 128x128 input with 4 scalar transform parameter prediction.

ScaleOffsetDetector is a lightweight CNN with ~1.4M parameters.
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

# Import our modules
from scale_offset_model import ScaleOffsetDetector, ScaleOffsetLoss
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
        target_size=32,
        input_size=128, 
        transform_fraction=args.transform_fraction,
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
            target_size=32,
            input_size=128,
            transform_fraction=args.transform_fraction
        )
        datasets.append(hf_dataset)
    
    # Add local dataset if specified
    if args.source_images_dir:
        print(f"Loading local images from: {args.source_images_dir}")
        local_dataset = PixelArtDataset(
            source_images_dir=args.source_images_dir,
            num_samples=args.local_samples,
            synthesizer=synthesizer,
            target_size=32,
            input_size=128,
            transform_fraction=args.transform_fraction
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




def create_validation_image_grid(targets, inputs, pred_transforms, true_transforms, max_samples=16):
    """Create a grid of validation images for wandb logging.
    
    Enhanced to show:
    1. Ground truth target from dataset
    2. Input image (what the model sees)
    3. Target upscaled for comparison
    4. Predicted vs actual transform parameters comparison
    
    Args:
        targets: Ground truth target images (B, C, H, W)
        inputs: Input images (B, C, H, W)
        pred_transforms: Predicted transform parameters (B, 4)
        true_transforms: Ground truth transform parameters (B, 4)
        max_samples: Maximum number of samples to include
        
    Returns:
        wandb.Image objects for logging
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    batch_size = min(targets.size(0), max_samples)
    
    # Create a grid showing: ground truth target | input image | target upscaled | transform params
    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Images
        target_img = targets[i].cpu().permute(1, 2, 0).numpy()
        input_img = inputs[i].cpu().permute(1, 2, 0).numpy()
        
        # Clip values to [0, 1] for display
        target_img = np.clip(target_img, 0, 1)
        input_img = np.clip(input_img, 0, 1)
        
        # Extract transform parameters
        true_t = true_transforms[i].cpu().numpy()
        pred_t = pred_transforms[i].cpu().numpy()
        
        # Ground truth target
        axes[i, 0].imshow(target_img)
        axes[i, 0].set_title(f"Ground Truth Target\nSample {i}")
        axes[i, 0].axis('off')
        
        # Input image (what the model sees)
        axes[i, 1].imshow(input_img)
        transform_text = f"True: [{true_t[0]:.2f}, {true_t[1]:.2f}, {true_t[2]:.2f}, {true_t[3]:.2f}]"
        axes[i, 1].set_title(f"Input Image\n{transform_text}")
        axes[i, 1].axis('off')
        
        # Target upscaled for better visibility
        from scipy.ndimage import zoom
        if target_img.shape[0] < input_img.shape[0]:  # If target is smaller, upscale it
            scale_factor = input_img.shape[0] // target_img.shape[0]
            upscaled_target = zoom(target_img, (scale_factor, scale_factor, 1), order=0)
        else:
            upscaled_target = target_img
        axes[i, 2].imshow(upscaled_target)
        axes[i, 2].set_title(f"Target (Upscaled)\nfor Comparison")
        axes[i, 2].axis('off')
        
        # Transform parameters comparison
        axes[i, 3].axis('off')
        error = np.abs(pred_t - true_t)
        transform_text = f"Ground Truth Transform:\n"
        transform_text += f"  Scale X:  {true_t[0]:+6.3f}\n"
        transform_text += f"  Scale Y:  {true_t[1]:+6.3f}\n"
        transform_text += f"  Offset X: {true_t[2]:+6.3f}\n"
        transform_text += f"  Offset Y: {true_t[3]:+6.3f}\n\n"
        transform_text += f"Predicted Transform:\n"
        transform_text += f"  Scale X:  {pred_t[0]:+6.3f}\n"
        transform_text += f"  Scale Y:  {pred_t[1]:+6.3f}\n"
        transform_text += f"  Offset X: {pred_t[2]:+6.3f}\n"
        transform_text += f"  Offset Y: {pred_t[3]:+6.3f}\n\n"
        transform_text += f"Absolute Error:\n"
        transform_text += f"  Scale X:  {error[0]:6.3f}\n"
        transform_text += f"  Scale Y:  {error[1]:6.3f}\n"
        transform_text += f"  Offset X: {error[2]:6.3f}\n"
        transform_text += f"  Offset Y: {error[3]:6.3f}\n"
        transform_text += f"  Mean MAE: {error.mean():6.3f}"
        
        axes[i, 3].text(0.05, 0.95, transform_text, fontsize=9, family='monospace',
                       verticalalignment='top', transform=axes[i, 3].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        axes[i, 3].set_title(f"Transform Prediction")
    
    plt.tight_layout()
    
    # Convert to wandb Image
    wandb_image = wandb.Image(fig)
    plt.close(fig)
    
    return wandb_image


def train_epoch(model, dataloader, optimizer, criterion, device, epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_scale_loss = 0.0
    total_offset_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, target_data) in enumerate(progress_bar):
        inputs = inputs.to(device)
        # Extract transformation parameters from target data
        _, true_transform_params = target_data
        true_transform_params = true_transform_params.to(device)
        
        optimizer.zero_grad()
        pred_transform_params = model(inputs)
        
        # Compute loss
        loss, loss_dict = criterion(pred_transform_params, true_transform_params)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss_dict['total_loss']
        total_scale_loss += loss_dict['scale_loss']
        total_offset_loss += loss_dict['offset_loss']
        
        avg_loss = total_loss / (batch_idx + 1)
        avg_scale_loss = total_scale_loss / (batch_idx + 1)
        avg_offset_loss = total_offset_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.6f}',
            'scale': f'{avg_scale_loss:.6f}',
            'offset': f'{avg_offset_loss:.6f}'
        })
        
        # Log batch metrics to wandb
        if wandb.run is not None:
            wandb.log({
                'batch_loss': loss_dict['total_loss'],
                'batch_scale_loss': loss_dict['scale_loss'],
                'batch_offset_loss': loss_dict['offset_loss'],
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return {
        'train_loss': total_loss / num_batches,
        'train_scale_loss': total_scale_loss / num_batches,
        'train_offset_loss': total_offset_loss / num_batches
    }


def validate_epoch(model, dataloader, criterion, device, epoch: int,
                   log_images: bool = False, max_image_samples: int = 16) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_scale_loss = 0.0
    total_offset_loss = 0.0
    total_mae = 0.0
    total_scale_mae = 0.0
    total_offset_mae = 0.0
    num_batches = len(dataloader)
    
    # For image logging
    logged_samples = 0
    sample_inputs = []
    sample_targets = []
    sample_pred_transforms = []
    sample_true_transforms = []
    
    with torch.no_grad():
        for batch_idx, (inputs, target_data) in enumerate(tqdm(dataloader, desc=f'Val {epoch}')):
            inputs = inputs.to(device)
            # Extract target images and transformation parameters from target data
            target_images, true_transform_params = target_data
            target_images = target_images.to(device)
            true_transform_params = true_transform_params.to(device)
            
            pred_transform_params = model(inputs)
            
            # Compute losses
            loss, loss_dict = criterion(pred_transform_params, true_transform_params)
            mae = torch.abs(pred_transform_params - true_transform_params).mean()
            scale_mae = torch.abs(pred_transform_params[:, :2] - true_transform_params[:, :2]).mean()
            offset_mae = torch.abs(pred_transform_params[:, 2:] - true_transform_params[:, 2:]).mean()
            
            total_loss += loss_dict['total_loss']
            total_scale_loss += loss_dict['scale_loss']
            total_offset_loss += loss_dict['offset_loss']
            total_mae += mae.item()
            total_scale_mae += scale_mae.item()
            total_offset_mae += offset_mae.item()
            
            # Collect samples for image logging
            if log_images and logged_samples < max_image_samples and wandb.run is not None:
                samples_to_take = min(inputs.size(0), max_image_samples - logged_samples)
                sample_inputs.append(inputs[:samples_to_take].cpu())
                sample_targets.append(target_images[:samples_to_take].cpu())
                sample_pred_transforms.append(pred_transform_params[:samples_to_take].cpu())
                sample_true_transforms.append(true_transform_params[:samples_to_take].cpu())
                logged_samples += samples_to_take
    
    # Log validation images to wandb
    if log_images and logged_samples > 0 and wandb.run is not None:
        # Concatenate all collected samples
        all_inputs = torch.cat(sample_inputs, dim=0)
        all_targets = torch.cat(sample_targets, dim=0)
        all_pred_transforms = torch.cat(sample_pred_transforms, dim=0)
        all_true_transforms = torch.cat(sample_true_transforms, dim=0)
        
        # Create validation image grid showing targets and inputs
        validation_image = create_validation_image_grid(
            all_targets, all_inputs,
            all_pred_transforms, all_true_transforms,
            max_samples=logged_samples
        )
        
        # Log to wandb
        wandb.log({
            "validation_samples": validation_image,
            "epoch": epoch
        })
    
    return {
        'val_loss': total_loss / num_batches,
        'val_scale_loss': total_scale_loss / num_batches,
        'val_offset_loss': total_offset_loss / num_batches,
        'val_mae': total_mae / num_batches,
        'val_scale_mae': total_scale_mae / num_batches,
        'val_offset_mae': total_offset_mae / num_batches
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
    parser = argparse.ArgumentParser(description='Train ScaleOffsetDetector for transform parameter prediction')
    
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
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--transform_fraction', type=float, default=0.5,
                       help='Fraction of samples with applied transformations (0.0-1.0)')
    parser.add_argument('--scale_weight', type=float, default=1.0,
                       help='Weight for scale loss component')
    parser.add_argument('--offset_weight', type=float, default=1.0,
                       help='Weight for offset loss component')
    
    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training setup
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints-scale-offset',
                       help='Output directory for checkpoints')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='scale-offset',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable wandb logging')
    
    # Debug arguments
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
    model = ScaleOffsetDetector(
        input_size=128,
        dropout_rate=args.dropout
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,}")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = ScaleOffsetLoss(scale_weight=args.scale_weight, offset_weight=args.offset_weight)
    
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
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch,
                                   log_images=args.log_validation_images and not args.disable_wandb,
                                   max_image_samples=args.max_validation_images)
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        
        print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.6f}, "
              f"Val Loss: {val_metrics['val_loss']:.6f}, Val MAE: {val_metrics['val_mae']:.6f}, "
              f"Scale MAE: {val_metrics['val_scale_mae']:.6f}, Offset MAE: {val_metrics['val_offset_mae']:.6f}")
        
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
