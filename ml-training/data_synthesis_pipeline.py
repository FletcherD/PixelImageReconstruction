#!/usr/bin/env python3
"""
Data synthesis pipeline for pixel art recovery training data.

Creates pairs of (degraded_input, ground_truth) images by:
1. Random cropping 33x33 sections from source images
2. Extracting center 1x1 pixel as ground truth
3. Scaling 33x33 section by 2.0-8.0x with random interpolation
4. JPEG compression (quality 70-95)
5. Final scaling to 132x132 (input)
"""

import os
import random
import io
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset, Dataset as HFDataset


class PixelArtDataSynthesizer:
    """Synthesizes training data for pixel art recovery."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def random_crop_33x33(self, image: Image.Image) -> Image.Image:
        """Randomly crop a 33x33 section from the input image."""
        width, height = image.size
        
        if width < 33 or height < 33:
            # If image is too small, resize it first
            scale_factor = max(33 / width, 33 / height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.LANCZOS)
            width, height = image.size
        
        # Random crop coordinates
        left = random.randint(0, width - 33)
        top = random.randint(0, height - 33)
        
        return image.crop((left, top, left + 33, top + 33))
    
    def posterize_image(self, image: Image.Image) -> Image.Image:
        """Posterize image to random number of colors between 8 and 64."""
        num_colors = random.randint(4, 64)
        
        # Convert to P mode with adaptive palette
        quantized = image.quantize(colors=num_colors, method=Image.MEDIANCUT)
        # Convert back to RGB
        return quantized.convert('RGB')
    
    def random_scale(self, image: Image.Image) -> Image.Image:
        """Scale image by random factor 2.0-8.0x with random interpolation."""
        # Random scale factors (can be different for x and y)
        base_scale = random.uniform(2.0, 8.0)
        scale_x = base_scale * random.uniform(0.9, 1.1)  # Allow slight variation
        scale_y = base_scale * random.uniform(0.9, 1.1)
        
        # Random interpolation method
        interpolation_methods = [Image.NEAREST, Image.BILINEAR, Image.LANCZOS]
        interpolation = random.choice(interpolation_methods)
        
        width, height = image.size
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        
        return image.resize((new_width, new_height), interpolation)
    
    def apply_jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Apply JPEG compression with quality between 70-95."""
        quality = random.randint(70, 95)
        
        # Save to bytes buffer with JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load back from buffer
        return Image.open(buffer).convert('RGB')
    
    def extract_center_pixel(self, image: Image.Image) -> Image.Image:
        """Extract the center 1x1 pixel from a 33x33 image."""
        # Center pixel is at coordinates (16, 16) in a 33x33 image
        center_x, center_y = 16, 16
        return image.crop((center_x, center_y, center_x + 1, center_y + 1))
    
    def scale_to_final(self, image: Image.Image) -> Image.Image:
        """Scale image to final size."""
        return image.resize((128, 128), Image.LANCZOS)
    
    def synthesize_pair(self, source_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Create a (input, ground_truth) pair from source image.
        
        Returns:
            Tuple of (degraded_input_132x132, ground_truth_1x1)
        """
        # Step 1: Random crop 33x33
        cropped = self.random_crop_33x33(source_image)
        
        # Step 2: Extract center 1x1 pixel (this becomes our ground truth)
        ground_truth = self.extract_center_pixel(cropped)
        
        # Step 3: Scale up with random interpolation
        scaled = self.random_scale(cropped)
        
        # Step 4: JPEG compression
        compressed = self.apply_jpeg_compression(scaled)
        
        # Step 5: Scale to final (this becomes our input)
        input_image = self.scale_to_final(compressed)
        
        return input_image, ground_truth


class PixelArtDataset(Dataset):
    """PyTorch Dataset for pixel art recovery training."""
    
    def __init__(self, 
                 source_images_dir: str,
                 num_samples: int,
                 synthesizer: Optional[PixelArtDataSynthesizer] = None,
                 transform_input: Optional[transforms.Compose] = None,
                 transform_target: Optional[transforms.Compose] = None):
        """
        Args:
            source_images_dir: Directory containing source images
            num_samples: Number of synthetic samples to generate
            synthesizer: Data synthesizer (creates one if None)
            transform_input: Transforms for input images
            transform_target: Transforms for target images
        """
        self.source_images_dir = Path(source_images_dir)
        self.num_samples = num_samples
        self.synthesizer = synthesizer or PixelArtDataSynthesizer()
        self.transform_input = transform_input
        self.transform_target = transform_target
        
        # Get list of source images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.source_image_paths = [
            p for p in self.source_images_dir.rglob('*')
            if p.suffix.lower() in image_extensions
        ]
        
        if not self.source_image_paths:
            raise ValueError(f"No images found in {source_images_dir}")
        
        print(f"Found {len(self.source_image_paths)} source images")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Randomly select a source image
        source_path = random.choice(self.source_image_paths)
        
        try:
            source_image = Image.open(source_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {source_path}: {e}")
            # Fallback to first image
            source_image = Image.open(self.source_image_paths[0]).convert('RGB')
        
        # Synthesize input/target pair
        input_image, target_image = self.synthesizer.synthesize_pair(source_image)
        
        # Apply transforms
        if self.transform_input:
            input_tensor = self.transform_input(input_image)
        else:
            input_tensor = transforms.ToTensor()(input_image)
        
        if self.transform_target:
            target_tensor = self.transform_target(target_image)
        else:
            target_tensor = transforms.ToTensor()(target_image)
        
        return input_tensor, target_tensor


class HuggingFacePixelArtDataset(Dataset):
    """PyTorch Dataset for pixel art recovery training using HuggingFace datasets."""
    
    def __init__(self, 
                 dataset_name: str,
                 num_samples: int,
                 synthesizer: Optional[PixelArtDataSynthesizer] = None,
                 transform_input: Optional[transforms.Compose] = None,
                 transform_target: Optional[transforms.Compose] = None,
                 split: str = "train",
                 image_column: str = "image",
                 streaming: bool = False,
                 **load_dataset_kwargs):
        """
        Args:
            dataset_name: Name of the HuggingFace dataset
            num_samples: Number of synthetic samples to generate
            synthesizer: Data synthesizer (creates one if None)
            transform_input: Transforms for input images
            transform_target: Transforms for target images
            split: Dataset split to use (train, validation, test)
            image_column: Name of the column containing images
            streaming: Whether to use streaming mode for large datasets
            **load_dataset_kwargs: Additional arguments to pass to load_dataset
        """
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.synthesizer = synthesizer or PixelArtDataSynthesizer()
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.image_column = image_column
        self.streaming = streaming
        
        # Load the HuggingFace dataset
        print(f"Loading HuggingFace dataset: {dataset_name}")
        self.hf_dataset = load_dataset(
            dataset_name, 
            split=split, 
            streaming=streaming,
            **load_dataset_kwargs
        )
        
        if not streaming:
            print(f"Loaded {len(self.hf_dataset)} images from {dataset_name}")
            if len(self.hf_dataset) == 0:
                raise ValueError(f"No images found in dataset {dataset_name}")
        else:
            print(f"Loaded streaming dataset {dataset_name}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Randomly select a source image from the dataset
        if self.streaming:
            # For streaming datasets, we need to iterate to get a random sample
            # This is less efficient but necessary for streaming mode
            sample_idx = random.randint(0, 10000)  # Assume large dataset
            for i, sample in enumerate(self.hf_dataset):
                if i == sample_idx:
                    break
            else:
                # If we didn't find the sample, take the first one
                sample = next(iter(self.hf_dataset))
        else:
            # For non-streaming datasets, we can directly index
            sample_idx = random.randint(0, len(self.hf_dataset) - 1)
            sample = self.hf_dataset[sample_idx]
        
        # Extract the image from the sample
        source_image = sample[self.image_column]
        
        # Ensure it's a PIL Image
        if not isinstance(source_image, Image.Image):
            if hasattr(source_image, 'convert'):
                source_image = source_image.convert('RGB')
            else:
                # Try to convert array-like data to PIL Image
                source_image = Image.fromarray(np.array(source_image)).convert('RGB')
        else:
            source_image = source_image.convert('RGB')
        
        # Synthesize input/target pair
        input_image, target_image = self.synthesizer.synthesize_pair(source_image)
        
        # Apply transforms
        if self.transform_input:
            input_tensor = self.transform_input(input_image)
        else:
            input_tensor = transforms.ToTensor()(input_image)
        
        if self.transform_target:
            target_tensor = self.transform_target(target_image)
        else:
            target_tensor = transforms.ToTensor()(target_image)
        
        return input_tensor, target_tensor


def create_hf_dataset(dataset_name: str,
                     num_samples: int,
                     save_examples: bool = True,
                     examples_dir: str = "dataset_examples",
                     split: str = "train",
                     image_column: str = "image",
                     streaming: bool = False,
                     **load_dataset_kwargs) -> HuggingFacePixelArtDataset:
    """
    Create a HuggingFacePixelArtDataset and optionally save some examples.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        num_samples: Number of samples in the dataset
        save_examples: Whether to save example pairs
        examples_dir: Directory to save examples
        split: Dataset split to use (train, validation, test)
        image_column: Name of the column containing images
        streaming: Whether to use streaming mode for large datasets
        **load_dataset_kwargs: Additional arguments to pass to load_dataset
        
    Returns:
        HuggingFacePixelArtDataset instance
    """
    # Standard transforms for training
    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    target_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = HuggingFacePixelArtDataset(
        dataset_name=dataset_name,
        num_samples=num_samples,
        transform_input=input_transform,
        transform_target=target_transform,
        split=split,
        image_column=image_column,
        streaming=streaming,
        **load_dataset_kwargs
    )
    
    if save_examples:
        os.makedirs(examples_dir, exist_ok=True)
        synthesizer = PixelArtDataSynthesizer(seed=69)  # Fixed seed for reproducible examples
        
        print(f"Saving example pairs to {examples_dir}/")
        for i in range(num_samples):
            # Get a sample from the dataset
            sample_idx = random.randint(0, len(dataset.hf_dataset) - 1) if not streaming else i
            if streaming:
                # For streaming, just take next samples
                sample = next(iter(dataset.hf_dataset))
            else:
                sample = dataset.hf_dataset[sample_idx]
            
            source_image = sample[image_column]
            if not isinstance(source_image, Image.Image):
                if hasattr(source_image, 'convert'):
                    source_image = source_image.convert('RGB')
                else:
                    source_image = Image.fromarray(np.array(source_image)).convert('RGB')
            else:
                source_image = source_image.convert('RGB')
            
            input_img, target_img = synthesizer.synthesize_pair(source_image)
            
            # Save the pair
            input_img.save(f"{examples_dir}/hf_example_{i:02d}_input.png")
            target_img.save(f"{examples_dir}/hf_example_{i:02d}_target.png")
            
    
    return dataset


def create_dataset(source_images_dir: str, 
                  num_samples: int, 
                  save_examples: bool = True,
                  examples_dir: str = "dataset_examples") -> PixelArtDataset:
    """
    Create a PixelArtDataset and optionally save some examples.
    
    Args:
        source_images_dir: Directory containing source images
        num_samples: Number of samples in the dataset
        save_examples: Whether to save example pairs
        examples_dir: Directory to save examples
        
    Returns:
        PixelArtDataset instance
    """
    # Standard transforms for training
    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    target_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = PixelArtDataset(
        source_images_dir=source_images_dir,
        num_samples=num_samples,
        transform_input=input_transform,
        transform_target=target_transform
    )
    
    if save_examples:
        os.makedirs(examples_dir, exist_ok=True)
        synthesizer = PixelArtDataSynthesizer(seed=42)  # Fixed seed for reproducible examples
        
        print(f"Saving example pairs to {examples_dir}/")
        for i in range(num_samples):  # Save examples
            source_path = random.choice(dataset.source_image_paths)
            source_image = Image.open(source_path).convert('RGB')
            
            input_img, target_img = synthesizer.synthesize_pair(source_image)
            
            # Save the pair
            input_img.save(f"{examples_dir}/example_{i:02d}_input.png")
            target_img.save(f"{examples_dir}/example_{i:02d}_target.png")
    
    return dataset


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create pixel art recovery training dataset")
    parser.add_argument("--source_dir", type=str,
                       help="Directory containing source images")
    parser.add_argument("--hf_dataset", type=str,
                       help="HuggingFace dataset name (e.g., 'imagenet-1k')")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to generate")
    parser.add_argument("--examples_dir", type=str, default="dataset_examples",
                       help="Directory to save example pairs")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (for HF datasets)")
    parser.add_argument("--image_column", type=str, default="image",
                       help="Column name containing images (for HF datasets)")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming mode for large HF datasets")
    
    args = parser.parse_args()
    
    # Check that exactly one source is provided
    if bool(args.source_dir) == bool(args.hf_dataset):
        parser.error("Provide exactly one of --source_dir or --hf_dataset")
    
    if args.hf_dataset:
        print(f"Creating dataset with {args.num_samples} samples from HuggingFace dataset: {args.hf_dataset}")
        dataset = create_hf_dataset(
            dataset_name=args.hf_dataset,
            num_samples=args.num_samples,
            save_examples=True,
            examples_dir=args.examples_dir,
            split=args.split,
            image_column=args.image_column,
            streaming=args.streaming
        )
    else:
        print(f"Creating dataset with {args.num_samples} samples from {args.source_dir}")
        dataset = create_dataset(
            source_images_dir=args.source_dir,
            num_samples=args.num_samples,
            save_examples=True,
            examples_dir=args.examples_dir
        )
    
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    # Test loading a sample
    input_tensor, target_tensor = dataset[0]
    print(f"Input shape: {input_tensor.shape}")   # Should be [3, 132, 132]
    print(f"Target shape: {target_tensor.shape}") # Should be [3, 1, 1]
