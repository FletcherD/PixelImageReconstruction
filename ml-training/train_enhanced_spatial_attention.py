#!/usr/bin/env python3
"""
Training script for enhanced spatial attention U-Net models.

Uses improved spatial attention with better initialization, temperature control,
and configurable attention strength for pixel art reconstruction.
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
from enhanced_spatial_attention_models import (
    EnhancedMediumUNetSpatialAttention64x16,
    ConfigurableAttentionUNet64x16
)
from medium_unet import count_parameters, mse_loss
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
        target_size=16,
        input_size=64,  # No transforms for simple patch reconstruction
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
            target_size=16,
            input_size=64,
        )
        datasets.append(hf_dataset)
    
    # Add local dataset if specified
    if args.source_images_dir:
        print(f"Loading local images from: {args.source_images_dir}")
        local_dataset = PixelArtDataset(
            source_images_dir=args.source_images_dir,
            num_samples=args.local_samples,
            synthesizer=synthesizer,
            target_size=16,
            input_size=64,
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


def create_validation_image_grid(inputs, targets, pred_images, max_samples=16):
    """Create a grid of validation images for wandb logging."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    batch_size = min(inputs.size(0), max_samples)
    
    # Create a grid showing: input center | target | prediction
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Extract center regions for visualization (16x16 from 64x64)
        input_center = inputs[i, :, 24:40, 24:40].cpu().permute(1, 2, 0).numpy()
        target_img = targets[i].cpu().permute(1, 2, 0).numpy()
        pred_img = pred_images[i].cpu().permute(1, 2, 0).numpy()
        
        # Clip values to [0, 1] for display
        input_center = np.clip(input_center, 0, 1)
        target_img = np.clip(target_img, 0, 1)
        pred_img = np.clip(pred_img, 0, 1)
        
        # Input center
        axes[i, 0].imshow(input_center)
        axes[i, 0].set_title(f"Input Center\nSample {i}")
        axes[i, 0].axis('off')
        
        # Target (ground truth)
        axes[i, 1].imshow(target_img)
        axes[i, 1].set_title(f"Target\n(Ground Truth)")
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred_img)
        axes[i, 2].set_title(f"Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Convert to wandb Image
    wandb_image = wandb.Image(fig)
    plt.close(fig)
    
    return wandb_image


def train_epoch(model, dataloader, optimizer, device, epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, target_data) in enumerate(progress_bar):
        inputs = inputs.to(device)
        # Extract target images from target data (ignore transform params)
        targets = target_data
        targets = targets.to(device)
        
        optimizer.zero_grad()
        mu = model(inputs)
        
        # Image reconstruction loss
        loss = mse_loss(mu, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.6f}'
        })
        
        # Log batch metrics to wandb
        if wandb.run is not None:
            wandb.log({
                'batch_loss': loss.item(),
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return {
        'train_loss': total_loss / num_batches
    }


def validate_epoch(model, dataloader, device, epoch: int,
                   log_images: bool = False, 
                   max_image_samples: int = 16) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = len(dataloader)
    
    # For image logging
    logged_samples = 0
    sample_inputs = []
    sample_targets = []
    sample_pred_images = []
    
    with torch.no_grad():
        for batch_idx, (inputs, target_data) in enumerate(tqdm(dataloader, desc=f'Val {epoch}')):
            inputs = inputs.to(device)
            # Extract target images from target data (ignore transform params)
            targets = target_data
            targets = targets.to(device)
            batch_size = inputs.size(0)
            
            mu = model(inputs)
            
            # Image losses
            loss = mse_loss(mu, targets)
            mae = torch.abs(mu - targets).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            
            # Collect samples for image logging
            if log_images and logged_samples < max_image_samples and wandb.run is not None:
                samples_to_take = min(batch_size, max_image_samples - logged_samples)
                sample_inputs.append(inputs[:samples_to_take].cpu())
                sample_targets.append(targets[:samples_to_take].cpu())
                sample_pred_images.append(mu[:samples_to_take].cpu())
                logged_samples += samples_to_take
    
    # Log validation images to wandb
    if log_images and logged_samples > 0 and wandb.run is not None:
        # Concatenate all collected samples
        all_inputs = torch.cat(sample_inputs, dim=0)
        all_targets = torch.cat(sample_targets, dim=0)
        all_pred_images = torch.cat(sample_pred_images, dim=0)
        
        # Create validation image grid
        validation_image = create_validation_image_grid(
            all_inputs, all_targets, all_pred_images,
            max_samples=logged_samples
        )
        
        # Log to wandb
        wandb.log({
            "validation_samples": validation_image,
            "epoch": epoch
        })
    
    return {
        'val_loss': total_loss / num_batches,
        'val_mae': total_mae / num_batches
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
    parser = argparse.ArgumentParser(description='Train Enhanced Spatial Attention U-Net for patch prediction')
    
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
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='enhanced',
                       choices=['enhanced', 'configurable'],
                       help='Type of enhanced attention model')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--use_transpose', action='store_true', default=True,
                       help='Use transposed convolution for upsampling')
    parser.add_argument('--final_activation', type=str, default='sigmoid',
                       choices=['sigmoid', 'tanh', 'relu', 'none'],
                       help='Final activation function')
    
    # Enhanced attention arguments
    parser.add_argument('--attention_type', type=str, default='improved',
                       choices=['improved', 'positional', 'multiscale'],
                       help='Type of spatial attention mechanism')
    parser.add_argument('--attention_strength', type=float, default=0.2,
                       help='Initial strength of attention (gamma parameter)')
    parser.add_argument('--attention_temperature', type=float, default=0.7,
                       help='Temperature for attention softmax (lower = sharper)')
    parser.add_argument('--attention_reduction', type=int, default=4,
                       help='Channel reduction factor for attention')
    parser.add_argument('--attention_locations', nargs='+', default=['bottleneck', 'decoder'],
                       choices=['bottleneck', 'decoder'],
                       help='Where to apply attention')
    
    # Training setup
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints-enhanced-spatial-attention',
                       help='Output directory for checkpoints')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='enhanced-spatial-attention',
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
    
    # Resume training arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory with attention config
    attention_suffix = f"{args.attention_type}-s{args.attention_strength}-t{args.attention_temperature}"
    output_dir = f"{args.output_dir}-{attention_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    # Initialize wandb
    if not args.disable_wandb:
        wandb_run_name = args.wandb_run_name or f"{args.attention_type}-s{args.attention_strength}-t{args.attention_temperature}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args)
        )
    
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
    if args.model_type == 'enhanced':
        model = EnhancedMediumUNetSpatialAttention64x16(
            dropout=args.dropout,
            use_transpose=args.use_transpose,
            final_activation=args.final_activation,
            attention_type=args.attention_type,
            attention_strength=args.attention_strength,
            attention_temperature=args.attention_temperature,
            attention_reduction=args.attention_reduction
        )
    elif args.model_type == 'configurable':
        model = ConfigurableAttentionUNet64x16(
            dropout=args.dropout,
            use_transpose=args.use_transpose,
            final_activation=args.final_activation,
            attention_type=args.attention_type,
            attention_locations=args.attention_locations,
            attention_strength=args.attention_strength,
            attention_temperature=args.attention_temperature,
            attention_reduction=args.attention_reduction
        )
    
    model = model.to(device)
    print(f"Model: {args.model_type} with {args.attention_type} attention")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Attention config: strength={args.attention_strength}, temp={args.attention_temperature}, reduction={args.attention_reduction}")
    
    # Log attention info if configurable model
    if hasattr(model, 'get_attention_info'):
        attention_info = model.get_attention_info()
        print(f"Attention parameters: {attention_info['parameters']:,}")
        print(f"Attention modules: {attention_info['modules']}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")
        
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}, best val loss: {best_val_loss:.6f}")
    
    # Log model architecture to wandb
    if not args.disable_wandb:
        wandb.watch(model, log_freq=100)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, epoch,
                                   log_images=args.log_validation_images and not args.disable_wandb,
                                   max_image_samples=args.max_validation_images)
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        
        print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.6f}, "
              f"Val Loss: {val_metrics['val_loss']:.6f}, Val MAE: {val_metrics['val_mae']:.6f}")
        
        if not args.disable_wandb:
            wandb.log(metrics)
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics['val_loss'],
                os.path.join(args.output_dir, 'best_model.pth')
            )
            print(f"  → New best model saved!")
        
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
    print(f"Models saved to: {args.output_dir}")
    
    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()