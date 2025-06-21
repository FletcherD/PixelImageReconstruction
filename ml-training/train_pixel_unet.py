#!/usr/bin/env python3
"""
Training script for PixelUNet model using the data synthesis pipeline.
Uses both local images and HuggingFace 'nerijs/pixelparti-128-v0.1' dataset.
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
import numpy as np
from tqdm import tqdm
import wandb

# Import our modules
from pixel_unet import PixelUNet, PixelUNetGAP, count_parameters
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


class PixelLoss(nn.Module):
    """Custom loss function for pixel prediction."""
    
    def __init__(self, loss_type: str = 'mse', color_weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.color_weight = color_weight
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Basic pixel loss
        pixel_loss = self.base_loss(pred, target)
        
        # Optional: Add color consistency loss
        if self.color_weight > 0:
            # Encourage color consistency across channels
            pred_mean = pred.mean(dim=1, keepdim=True)
            target_mean = target.mean(dim=1, keepdim=True)
            color_loss = self.base_loss(pred_mean, target_mean)
            pixel_loss = pixel_loss + self.color_weight * color_loss
        
        return pixel_loss


def create_datasets(args) -> tuple:
    """Create training and validation datasets."""
    synthesizer = PixelArtDataSynthesizer(seed=args.seed)
    
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
            input_size=128
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
            input_size=128
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Log batch metrics to wandb
        if wandb.run is not None:
            wandb.log({
                'batch_loss': loss.item(),
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return {'train_loss': total_loss / num_batches}


def validate_epoch(model, dataloader, criterion, device, epoch: int) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f'Val {epoch}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae = torch.abs(outputs - targets).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
    
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
    parser = argparse.ArgumentParser(description='Train PixelUNet for single pixel prediction')
    
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
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
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
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='mse',
                       choices=['mse', 'l1', 'smooth_l1'],
                       help='Loss function type')
    parser.add_argument('--color_weight', type=float, default=0.0,
                       help='Weight for color consistency loss')
    
    # Training setup
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='pixel-unet',
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
        model = PixelUNet(
            dropout=args.dropout,
            use_transpose=args.use_transpose,
            final_activation=args.final_activation
        )
    else:  # gap
        model = PixelUNetGAP(
            dropout=args.dropout,
            use_transpose=args.use_transpose,
            final_activation=args.final_activation
        )
    
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create loss function and optimizer
    criterion = PixelLoss(args.loss_type, args.color_weight)
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
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
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
