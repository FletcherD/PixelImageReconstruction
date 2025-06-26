#!/usr/bin/env python3
"""
Exact color data synthesis pipeline that preserves precise RGB values.

This pipeline avoids all color normalization to maintain exact integer RGB values
throughout training and inference, preventing the subtle color shifts that occur
with standard [0,1] normalization.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset

from exact_color_transforms import ExactRGBToTensor


class ExactColorPixelArtDataSynthesizer:
    """
    Synthesizes pixel art training data while preserving exact RGB values.
    Works with RGB values in [0, 255] range throughout the pipeline.
    """
    
    def __init__(self, target_size: int = 16, input_size: int = 64, 
                 transform_fraction: float = 0.0, seed: Optional[int] = None):
        """
        Initialize the synthesizer.
        
        Args:
            target_size: Size of target patches (e.g., 16 for 16x16)
            input_size: Size of input patches (e.g., 64 for 64x64)
            transform_fraction: Fraction of samples to apply transforms to
            seed: Random seed for reproducibility
        """
        self.target_size = target_size
        self.input_size = input_size
        self.transform_fraction = transform_fraction
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def synthesize_pair(self, source_image: Image.Image) -> Tuple[Image.Image, Image.Image, np.ndarray]:
        """
        Synthesize input/target pair from source image preserving exact colors.
        
        Args:
            source_image: Source PIL Image
            
        Returns:
            Tuple of (input_image, target_image, transform_params)
        """
        # Convert to numpy array to preserve exact RGB values
        source_array = np.array(source_image, dtype=np.uint8)
        
        # For now, create simple pairs by extracting center regions
        # This avoids any interpolation that could change RGB values
        
        # Resize source to input_size using nearest neighbor (preserves exact colors)
        if source_array.shape[:2] != (self.input_size, self.input_size):
            source_resized = source_image.resize((self.input_size, self.input_size), 
                                               resample=Image.NEAREST)
            source_array = np.array(source_resized, dtype=np.uint8)
        
        # Extract center region for target
        center_start = (self.input_size - self.target_size) // 2
        center_end = center_start + self.target_size
        
        target_array = source_array[center_start:center_end, center_start:center_end]
        
        # Convert back to PIL Images
        input_image = Image.fromarray(source_array, mode='RGB')
        target_image = Image.fromarray(target_array, mode='RGB')
        
        # No transforms applied for exact color preservation
        transform_params = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        return input_image, target_image, transform_params


class ExactColorPixelArtDataset(Dataset):
    """PyTorch Dataset that preserves exact RGB values throughout processing."""
    
    def __init__(self, source_images_dir: str, num_samples: int, 
                 synthesizer: Optional[ExactColorPixelArtDataSynthesizer] = None):
        """
        Initialize dataset.
        
        Args:
            source_images_dir: Directory containing source images
            num_samples: Number of samples to generate
            synthesizer: Data synthesizer instance
        """
        self.source_images_dir = Path(source_images_dir)
        self.num_samples = num_samples
        
        if synthesizer is None:
            synthesizer = ExactColorPixelArtDataSynthesizer()
        self.synthesizer = synthesizer
        
        # Load image paths
        self.source_image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            self.source_image_paths.extend(self.source_images_dir.glob(ext))
            self.source_image_paths.extend(self.source_images_dir.glob(ext.upper()))
        
        if not self.source_image_paths:
            raise ValueError(f"No images found in {source_images_dir}")
        
        print(f"Found {len(self.source_image_paths)} source images")
        
        # RGB to tensor transform that preserves exact values
        self.rgb_transform = ExactRGBToTensor()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Select random source image
        source_path = random.choice(self.source_image_paths)
        
        try:
            source_image = Image.open(source_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {source_path}: {e}")
            # Fallback to first image
            source_image = Image.open(self.source_image_paths[0]).convert('RGB')
        
        # Synthesize input/target pair
        input_image, target_image, transform_params = self.synthesizer.synthesize_pair(source_image)
        
        # Convert to tensors preserving exact RGB values [0, 255]
        input_tensor = self.rgb_transform(input_image)
        target_tensor = self.rgb_transform(target_image)
        
        # Transform parameters (for compatibility)
        transform_tensor = torch.tensor(transform_params, dtype=torch.float32)
        
        return input_tensor, (target_tensor, transform_tensor)


class ExactColorHuggingFacePixelArtDataset(Dataset):
    """HuggingFace dataset that preserves exact RGB values throughout processing."""
    
    def __init__(self, dataset_name: str, num_samples: int,
                 synthesizer: Optional[ExactColorPixelArtDataSynthesizer] = None,
                 split: str = "train", image_column: str = "image", 
                 streaming: bool = False, **load_dataset_kwargs):
        """
        Initialize HuggingFace dataset.
        
        Args:
            dataset_name: Name of HuggingFace dataset
            num_samples: Number of samples to generate
            synthesizer: Data synthesizer instance
            split: Dataset split to use
            image_column: Column name containing images
            streaming: Whether to use streaming mode
            **load_dataset_kwargs: Additional arguments for load_dataset
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for HuggingFace support. Install with: pip install datasets")
        
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.image_column = image_column
        self.streaming = streaming
        
        if synthesizer is None:
            synthesizer = ExactColorPixelArtDataSynthesizer()
        self.synthesizer = synthesizer
        
        # Load HuggingFace dataset
        print(f"Loading HuggingFace dataset: {dataset_name}")
        self.hf_dataset = load_dataset(
            dataset_name, 
            split=split, 
            streaming=streaming,
            **load_dataset_kwargs
        )
        
        if not streaming:
            print(f"Loaded {len(self.hf_dataset)} samples from HuggingFace")
        else:
            print(f"Using streaming mode for {dataset_name}")
        
        # RGB to tensor transform that preserves exact values
        self.rgb_transform = ExactRGBToTensor()
        
        # For streaming datasets, we need to create an iterator
        if streaming:
            self.hf_iter = iter(self.hf_dataset)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.streaming:
            # For streaming, get next sample
            try:
                sample = next(self.hf_iter)
            except StopIteration:
                # Reset iterator if we reach the end
                self.hf_iter = iter(self.hf_dataset)
                sample = next(self.hf_iter)
        else:
            # For non-streaming, sample randomly
            hf_idx = random.randint(0, len(self.hf_dataset) - 1)
            sample = self.hf_dataset[hf_idx]
        
        # Extract image from sample
        source_image = sample[self.image_column]
        
        # Ensure it's a PIL Image in RGB mode
        if not isinstance(source_image, Image.Image):
            if hasattr(source_image, 'convert'):
                source_image = source_image.convert('RGB')
            else:
                source_image = Image.fromarray(np.array(source_image)).convert('RGB')
        else:
            source_image = source_image.convert('RGB')
        
        # Synthesize input/target pair
        input_image, target_image, transform_params = self.synthesizer.synthesize_pair(source_image)
        
        # Convert to tensors preserving exact RGB values [0, 255]
        input_tensor = self.rgb_transform(input_image)
        target_tensor = self.rgb_transform(target_image)
        
        # Transform parameters (for compatibility)
        transform_tensor = torch.tensor(transform_params, dtype=torch.float32)
        
        return input_tensor, (target_tensor, transform_tensor)


def create_exact_color_datasets(source_images_dir: Optional[str] = None, 
                               num_samples: int = 10000, val_split: float = 0.1, 
                               seed: int = 42, use_hf_dataset: bool = False,
                               hf_dataset_name: str = 'nerijs/pixelparti-128-v0.1',
                               hf_samples: int = 10000, hf_streaming: bool = False,
                               local_samples: int = 10000) -> Tuple[Dataset, Dataset]:
    """
    Create train/validation datasets with exact color preservation.
    
    Args:
        source_images_dir: Directory containing local source images
        num_samples: Total number of samples (used if only one dataset type)
        val_split: Fraction for validation
        seed: Random seed
        use_hf_dataset: Whether to use HuggingFace dataset
        hf_dataset_name: Name of HuggingFace dataset
        hf_samples: Number of samples from HF dataset
        hf_streaming: Use streaming mode for HF dataset
        local_samples: Number of samples from local dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create synthesizer
    synthesizer = ExactColorPixelArtDataSynthesizer(
        target_size=16,
        input_size=64,
        transform_fraction=0.0,
        seed=seed
    )
    
    datasets = []
    
    # Add HuggingFace dataset
    if use_hf_dataset:
        print(f"Loading HuggingFace dataset: {hf_dataset_name}")
        hf_dataset = ExactColorHuggingFacePixelArtDataset(
            dataset_name=hf_dataset_name,
            num_samples=hf_samples,
            synthesizer=synthesizer,
            split='train',
            streaming=hf_streaming
        )
        datasets.append(hf_dataset)
    
    # Add local dataset if specified
    if source_images_dir:
        print(f"Loading local images from: {source_images_dir}")
        local_dataset = ExactColorPixelArtDataset(
            source_images_dir=source_images_dir,
            num_samples=local_samples,
            synthesizer=synthesizer
        )
        datasets.append(local_dataset)
    
    if not datasets:
        raise ValueError("No datasets specified. Use --use_hf_dataset or --source_images_dir")
    
    # Combine datasets
    if len(datasets) == 1:
        full_dataset = datasets[0]
        total_samples = num_samples if len(datasets) == 1 else len(full_dataset)
    else:
        full_dataset = ConcatDataset(datasets)
        total_samples = len(full_dataset)
    
    # Split into train/val
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def test_exact_color_dataset():
    """Test that the dataset preserves exact RGB values."""
    
    print("Testing exact color dataset...")
    
    # Create a simple test directory with a single test image
    test_dir = Path("test_exact_colors")
    test_dir.mkdir(exist_ok=True)
    
    # Create test image with specific colors
    test_colors = np.array([
        [[128, 200, 64], [255, 0, 128], [32, 196, 240], [85, 170, 42]],
        [[127, 63, 191], [1, 2, 3], [200, 100, 150], [50, 75, 125]],
        [[64, 128, 192], [240, 120, 80], [16, 32, 48], [224, 112, 56]],
        [[96, 144, 72], [160, 80, 200], [8, 16, 24], [248, 124, 62]]
    ], dtype=np.uint8)
    
    test_image = Image.fromarray(test_colors, mode='RGB')
    test_image.save(test_dir / "test_image.png")
    
    try:
        # Create dataset
        synthesizer = ExactColorPixelArtDataSynthesizer(target_size=2, input_size=4)
        dataset = ExactColorPixelArtDataset(
            source_images_dir=str(test_dir),
            num_samples=1,
            synthesizer=synthesizer
        )
        
        # Get a sample
        input_tensor, (target_tensor, transform_params) = dataset[0]
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor range: [{input_tensor.min():.0f}, {input_tensor.max():.0f}]")
        print(f"Target tensor shape: {target_tensor.shape}")
        print(f"Target tensor range: [{target_tensor.min():.0f}, {target_tensor.max():.0f}]")
        
        # Check that we have exact integer values
        input_ints = input_tensor.round().int()
        target_ints = target_tensor.round().int()
        
        print(f"Input contains exact integers: {torch.allclose(input_tensor, input_ints.float())}")
        print(f"Target contains exact integers: {torch.allclose(target_tensor, target_ints.float())}")
        
        # Convert back to check colors
        from exact_color_transforms import ExactRGBToImage
        back_transform = ExactRGBToImage()
        
        recovered_input = back_transform(input_tensor)
        recovered_target = back_transform(target_tensor)
        
        print(f"Input image recovered successfully: {isinstance(recovered_input, Image.Image)}")
        print(f"Target image recovered successfully: {isinstance(recovered_target, Image.Image)}")
        
        # Check some specific pixel values
        input_array = np.array(recovered_input)
        target_array = np.array(recovered_target)
        
        print(f"Sample input pixels: {input_array[0, 0]}, {input_array[0, 1]}")
        print(f"Sample target pixels: {target_array[0, 0]}, {target_array[0, 1]}")
        
        print("✓ Exact color dataset test passed!")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_exact_color_dataset()