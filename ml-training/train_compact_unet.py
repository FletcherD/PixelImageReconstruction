#!/usr/bin/env python3
"""
Training script for CompactUNet model using the data synthesis pipeline.
Uses both local images and HuggingFace 'nerijs/pixelparti-128-v0.1' dataset.
Configured for 128x128 input patches with 32x32 center patch prediction.
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
from compact_unet import CompactUNet, CompactUNetGAP, count_parameters, heteroscedastic_loss
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
    """Apply spatial transformation to input image based on parameters."""
    x_scale, y_scale, x_offset, y_offset = transform_params
    
    # Create affine transformation matrix
    theta = torch.zeros(1, 2, 3, device=image.device, dtype=image.dtype)
    theta[0, 0, 0] = x_scale
    theta[0, 1, 1] = y_scale
    theta[0, 0, 2] = x_offset
    theta[0, 1, 2] = y_offset
    
    # Apply transformation
    grid = torch.nn.functional.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
    transformed = torch.nn.functional.grid_sample(image.unsqueeze(0), grid, align_corners=False)
    
    return transformed.squeeze(0)


def generate_transform_params(batch_size, device):
    """Generate random transformation parameters for training."""
    # Half the samples perfectly aligned, half transformed
    aligned_samples = batch_size // 2
    transformed_samples = batch_size - aligned_samples
    
    # Aligned samples (all zeros)
    aligned_params = torch.zeros(aligned_samples, 4, device=device)
    
    # Transformed samples
    if transformed_samples > 0:
        # Random shift in circular pattern (up to 2 pixels)
        angles = torch.rand(transformed_samples, device=device) * 2 * math.pi
        distances = torch.rand(transformed_samples, device=device) * 2  # 0-2 pixels
        x_shift_pixels = distances * torch.cos(angles)
        y_shift_pixels = distances * torch.sin(angles)
        
        # Convert pixel shifts to normalized coordinates (-1 to 1 range for 128x128 input)
        x_offset = x_shift_pixels / 64  # 2 pixels / 64 = max range
        y_offset = y_shift_pixels / 64
        
        # Random scale (0.8 to 1.2)
        x_scale = torch.rand(transformed_samples, device=device) * 0.4 + 0.8  # [0.8, 1.2]
        y_scale = torch.rand(transformed_samples, device=device) * 0.4 + 0.8  # [0.8, 1.2]
        
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


def train_epoch(model, dataloader, optimizer, device, epoch: int) -> Dict[str, float]:
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
        true_transform_params = generate_transform_params(batch_size, device)
        
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
        mu, sigma_sq, pred_transform_params = model(transformed_inputs)
        
        # Image reconstruction loss
        img_loss = heteroscedastic_loss(mu, sigma_sq, targets)
        
        # Transform parameter loss
        trans_loss = transform_loss(pred_transform_params, true_transform_params)
        
        # Combined loss
        loss = img_loss + 0.1 * trans_loss  # Weight transform loss lower
        
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


def validate_epoch(model, dataloader, device, epoch: int) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_img_loss = 0.0
    total_transform_loss = 0.0
    total_mae = 0.0
    total_uncertainty = 0.0
    total_transform_mae = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f'Val {epoch}'):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # Generate random transformation parameters
            true_transform_params = generate_transform_params(batch_size, device)
            
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
            
            mu, sigma_sq, pred_transform_params = model(transformed_inputs)
            
            # Image losses
            img_loss = heteroscedastic_loss(mu, sigma_sq, targets)
            mae = torch.abs(mu - targets).mean()
            uncertainty = sigma_sq.mean()
            
            # Transform losses
            trans_loss = transform_loss(pred_transform_params, true_transform_params)
            transform_mae = torch.abs(pred_transform_params - true_transform_params).mean()
            
            # Combined loss
            loss = img_loss + 0.1 * trans_loss
            
            total_loss += loss.item()
            total_img_loss += img_loss.item()
            total_transform_loss += trans_loss.item()
            total_mae += mae.item()
            total_uncertainty += uncertainty.item()
            total_transform_mae += transform_mae.item()
    
    return {
        'val_loss': total_loss / num_batches,
        'val_img_loss': total_img_loss / num_batches,
        'val_transform_loss': total_transform_loss / num_batches,
        'val_mae': total_mae / num_batches,
        'val_uncertainty': total_uncertainty / num_batches,
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
    parser = argparse.ArgumentParser(description='Train CompactUNet for single pixel prediction')
    
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
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='standard', 
                       choices=['standard', 'gap'],
                       help='Model type: standard or gap (Global Average Pooling)')
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
    parser.add_argument('--output_dir', type=str, default='checkpoints-compact-unet',
                       help='Output directory for checkpoints')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='compact-unet',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable wandb logging')
    
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
    if args.model_type == 'standard':
        model = CompactUNet(
            dropout=args.dropout,
            use_transpose=args.use_transpose,
            final_activation=args.final_activation
        )
    else:  # gap
        model = CompactUNetGAP(
            dropout=args.dropout,
            final_activation=args.final_activation
        )
    
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
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
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        
        print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.6f}, "
              f"Val Loss: {val_metrics['val_loss']:.6f}, Val MAE: {val_metrics['val_mae']:.6f}, "
              f"Val Uncertainty: {val_metrics['val_uncertainty']:.6f}, "
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