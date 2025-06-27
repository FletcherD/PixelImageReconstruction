#!/usr/bin/env python3
"""
Data synthesis pipeline for pixel art recovery training data.

Creates pairs of (degraded_input, ground_truth) images by:
1. Random cropping crop_size x crop_size sections from source images
2. Extracting center 1x1 pixel as ground truth
3. Scaling cropped section by 2.0-8.0x with random interpolation
4. JPEG compression (quality 70-95)
5. Final scaling to input_size x input_size (input)
"""

import os
import random
import io
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import math
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from datasets import load_dataset, Dataset as HFDataset
from palette_utils import posterize_image_to_palette, palette_indices_to_one_hot


class PixelArtDataSynthesizer:
    """Synthesizes training data for pixel art recovery."""
    
    def __init__(self, 
                 target_size: int,
                 input_size: int,
                 complexity_threshold: float = 10.0,
                 max_retries: int = 10,
                 palette: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Args:
            target_size: Size of the target ground truth image (e.g., 32 for 32x32 target)
            input_size: Size of the final input image
            complexity_threshold: Minimum standard deviation for image complexity
            max_retries: Maximum attempts to find a complex enough crop
            palette: Optional palette for posterization (N, 3) RGB values [0, 255]
            seed: Random seed for reproducibility
        """        
        self.target_size = target_size
        self.input_size = input_size
        self.complexity_threshold = complexity_threshold
        self.max_retries = max_retries
        self.palette = palette
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def random_crop(self, image: Image.Image) -> Image.Image:
        """Randomly crop a target_size x target_size section from the input image."""
        width, height = image.size
        
        # Random crop coordinates
        left = random.randint(0, width - self.target_size)
        top = random.randint(0, height - self.target_size)
        
        return image.crop((left, top, left + self.target_size, top + self.target_size))
    
    def is_image_too_simple(self, image: Image.Image, threshold: Optional[float] = None) -> bool:
        """Check if image has too little variation (almost all one color)."""
        if threshold is None:
            threshold = self.complexity_threshold
            
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate standard deviation across all channels
        std_dev = np.std(img_array)
        
        # Return True if std dev is below threshold (too simple)
        return std_dev < threshold
    
    def posterize_image(self, image: Image.Image) -> Image.Image:
        """Posterize image to random number of colors between 8 and 64."""
        num_colors = random.randint(4, 64)
        
        # Convert to P mode with adaptive palette
        quantized = image.quantize(colors=num_colors, method=Image.MEDIANCUT)
        # Convert back to RGB
        return quantized.convert('RGB')
    
    def scale_pipeline(self, image: Image.Image) -> Image.Image:
        """Apply scaling pipeline: 2x nearest neighbor, then random 1x-4x scaling with random interpolation."""
        width, height = image.size
        
        # Step 1: Scale to 2x with nearest neighbor
        intermediate_size = (width * 2, height * 2)
        resize_2x = v2.Resize(intermediate_size, interpolation=v2.InterpolationMode.NEAREST)
        scaled_2x = resize_2x(image)
        
        # Step 2: Random scale between 1x and 4x with random interpolation
        scale_factor = random.uniform(1.0, 4.0)
        final_width = int(intermediate_size[0] * scale_factor * random.uniform(0.9,1.1))
        final_height = int(intermediate_size[1] * scale_factor * random.uniform(0.9,1.1))
        
        # Choose random interpolation method
        interpolation_method = random.choice([
            v2.InterpolationMode.NEAREST, 
            v2.InterpolationMode.BILINEAR, 
            v2.InterpolationMode.BICUBIC
        ])
        
        final_resize = v2.Resize((final_height, final_width), interpolation=interpolation_method)
        return final_resize(scaled_2x)
    
    def apply_jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Apply JPEG compression with quality between 85-99 using torchvision v2.JPEG."""
        quality = random.randint(85, 99)
        jpeg_transform = v2.JPEG(quality=(quality, quality))
        return jpeg_transform(image)
    
    
    def scale_to_final(self, image: Image.Image) -> Image.Image:
        """Scale image to final input_size using bilinear interpolation."""
        resize_transform = v2.Resize((self.input_size, self.input_size), interpolation=v2.InterpolationMode.BILINEAR)
        return resize_transform(image)
    

    
    def synthesize_pair(self, source_image: Image.Image) -> Tuple[Image.Image, Union[Image.Image, np.ndarray]]:
        """
        Create a (input, ground_truth) pair from source image.
        
        Pipeline:
        1. Random crop target_size x target_size region (ground truth)
        2. Scale to 2x with nearest neighbor
        3. Random scale 1x-4x with random interpolation (nearest/bilinear/bicubic)  
        4. JPEG compression (quality 85-99)
        5. Scale to final input_size with bilinear (input)
        
        Returns:
            Tuple of (degraded_input, ground_truth_data)
            - degraded_input: PIL Image for input
            - ground_truth_data: PIL Image (normal mode) or numpy array of indices (palette mode)
        """
        # Step 1: Random crop with complexity checking
        for attempt in range(self.max_retries):
            ground_truth = self.random_crop(source_image)
            ground_truth = self.posterize_image(ground_truth)
            
            # Check if the cropped region is too simple
            if not self.is_image_too_simple(ground_truth):
                break
        else:
            # If all attempts failed, use the last crop anyway
            # This prevents infinite loops with very simple source images
            pass

        # Step 2-3: Apply scaling pipeline (2x nearest + random 1x-4x scaling)
        scaled = self.scale_pipeline(ground_truth)
        
        # Step 4: JPEG compression
        compressed = self.apply_jpeg_compression(scaled)
        
        # Step 5: Scale to final input size with bilinear
        input_image = self.scale_to_final(compressed)
        
        # Step 6: Handle palette mode if palette is provided
        if self.palette is not None:
            # Posterize ground truth to palette and return indices
            _, ground_truth_indices = posterize_image_to_palette(ground_truth, self.palette)
            return input_image, ground_truth_indices
        else:
            # Return normal RGB ground truth
            return input_image, ground_truth


class PixelArtDataset(Dataset):
    """PyTorch Dataset for pixel art recovery training."""
    
    def __init__(self, 
                 source_images_dir: str,
                 num_samples: int,
                 target_size: int,
                 input_size: int,
                 synthesizer: Optional[PixelArtDataSynthesizer] = None,
                 transform_input: Optional[transforms.Compose] = None,
                 transform_target: Optional[transforms.Compose] = None,
                 palette: Optional[np.ndarray] = None
                 ):
        """
        Args:
            source_images_dir: Directory containing source images
            num_samples: Number of synthetic samples to generate
            synthesizer: Data synthesizer (creates one if None)
            transform_input: Transforms for input images
            transform_target: Transforms for target images
            target_size: Size of the target ground truth image
            input_size: Size of the final input image
            palette: Optional palette for posterization mode
        """
        self.source_images_dir = Path(source_images_dir)
        self.num_samples = num_samples
        self.synthesizer = synthesizer or PixelArtDataSynthesizer(
            target_size=target_size,
            input_size=input_size, 
            palette=palette
        )
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
        input_image, target_data = self.synthesizer.synthesize_pair(source_image)
        
        # Apply transforms
        if self.transform_input:
            input_tensor = self.transform_input(input_image)
        else:
            input_tensor = transforms.ToTensor()(input_image)
        
        # Handle palette mode vs regular mode
        if isinstance(target_data, np.ndarray):
            # Palette mode - target_data is indices array
            target_tensor = torch.from_numpy(target_data).long()
        else:
            # Regular mode - target_data is PIL Image
            if self.transform_target:
                target_tensor = self.transform_target(target_data)
            else:
                target_tensor = transforms.ToTensor()(target_data)
        
        # Return input tensor and target tensor
        return input_tensor, target_tensor


class HuggingFacePixelArtDataset(Dataset):
    """PyTorch Dataset for pixel art recovery training using HuggingFace datasets."""
    
    def __init__(self, 
                 dataset_name: str,
                 num_samples: int,
                 target_size: int,
                 input_size: int,
                 synthesizer: Optional[PixelArtDataSynthesizer] = None,
                 transform_input: Optional[transforms.Compose] = None,
                 transform_target: Optional[transforms.Compose] = None,
                 split: str = "train",
                 image_column: str = "image",
                 streaming: bool = False,
                 palette: Optional[np.ndarray] = None,
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
            target_size: Size of the target ground truth image
            input_size: Size of the final input image
            palette: Optional palette for posterization mode
            **load_dataset_kwargs: Additional arguments to pass to load_dataset
        """        
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.synthesizer = synthesizer or PixelArtDataSynthesizer(
            target_size=target_size,
            input_size=input_size,
            palette=palette
        )
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
        input_image, target_data = self.synthesizer.synthesize_pair(source_image)
        
        # Apply transforms
        if self.transform_input:
            input_tensor = self.transform_input(input_image)
        else:
            input_tensor = transforms.ToTensor()(input_image)
        
        # Handle palette mode vs regular mode
        if isinstance(target_data, np.ndarray):
            # Palette mode - target_data is indices array
            target_tensor = torch.from_numpy(target_data).long()
        else:
            # Regular mode - target_data is PIL Image
            if self.transform_target:
                target_tensor = self.transform_target(target_data)
            else:
                target_tensor = transforms.ToTensor()(target_data)
        
        # Return input tensor and target tensor
        return input_tensor, target_tensor


def create_hf_dataset(dataset_name: str,
                     num_samples: int,
                     target_size: int,
                     input_size: int,
                     save_examples: bool = True,
                     examples_dir: str = "dataset_examples",
                     split: str = "train",
                     image_column: str = "image",
                     streaming: bool = False,
                     palette: Optional[np.ndarray] = None,
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
        palette: Optional palette for posterization mode
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
        target_size=target_size,
        input_size=input_size,
        palette=palette,
        **load_dataset_kwargs
    )
    
    if save_examples:
        os.makedirs(examples_dir, exist_ok=True)
        synthesizer = PixelArtDataSynthesizer(target_size=target_size, input_size=input_size, palette=palette, seed=69)  # Fixed seed for reproducible examples
        
        print(f"Saving example pairs to {examples_dir}/")
        for i in range(min(10, num_samples)):  # Limit examples to 10
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
            
            # Save info
            with open(f"{examples_dir}/hf_example_{i:02d}_info.txt", "w") as f:
                f.write(f"Source size: {source_image.size}\n")
                f.write(f"Target size: {target_img.size if hasattr(target_img, 'size') else 'N/A'}\n")
                f.write(f"Input size: {input_img.size}\n")
            
    
    return dataset


def create_dataset(source_images_dir: str, 
                  num_samples: int, 
                  target_size: int,
                  input_size: int,
                  save_examples: bool = False,
                  examples_dir: str = "dataset_examples",
                  palette: Optional[np.ndarray] = None) -> PixelArtDataset:
    """
    Create a PixelArtDataset and optionally save some examples.
    
    Args:
        source_images_dir: Directory containing source images
        num_samples: Number of samples in the dataset
        save_examples: Whether to save example pairs
        examples_dir: Directory to save examples
        palette: Optional palette for posterization mode
        
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
        transform_target=target_transform,
        target_size=target_size,
        input_size=input_size,
        palette=palette
    )
    
    if save_examples:
        os.makedirs(examples_dir, exist_ok=True)
        synthesizer = PixelArtDataSynthesizer(target_size=target_size, input_size=input_size, palette=palette, seed=42)  # Fixed seed for reproducible examples
        
        print(f"Saving example pairs to {examples_dir}/")
        for i in range(min(10, num_samples)):  # Limit examples to 10
            source_path = random.choice(dataset.source_image_paths)
            source_image = Image.open(source_path).convert('RGB')
            
            input_img, target_img = synthesizer.synthesize_pair(source_image)
            
            # Save the pair
            input_img.save(f"{examples_dir}/example_{i:02d}_input.png")
            target_img.save(f"{examples_dir}/example_{i:02d}_target.png")
            
            # Save info
            with open(f"{examples_dir}/example_{i:02d}_info.txt", "w") as f:
                f.write(f"Source size: {source_image.size}\n")
                f.write(f"Target size: {target_img.size if hasattr(target_img, 'size') else 'N/A'}\n")
                f.write(f"Input size: {input_img.size}\n")
    
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
    parser.add_argument("--target_size", type=int, required=True,
                       help="Size of the target ground truth image")
    parser.add_argument("--input_size", type=int, required=True,
                       help="Size of the final input image")
    
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
            streaming=args.streaming,
            target_size=args.target_size,
            input_size=args.input_size
        )
    else:
        print(f"Creating dataset with {args.num_samples} samples from {args.source_dir}")
        dataset = create_dataset(
            source_images_dir=args.source_dir,
            num_samples=args.num_samples,
            save_examples=True,
            examples_dir=args.examples_dir,
            target_size=args.target_size,
            input_size=args.input_size
        )
    
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    # Test loading a sample
    input_tensor, target_tensor = dataset[0]
    print(f"Input shape: {input_tensor.shape}")   # Should be [3, input_size, input_size]
    print(f"Target shape: {target_tensor.shape}") # Should be [3, target_size, target_size]
