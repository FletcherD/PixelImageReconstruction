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
from datasets import load_dataset, Dataset as HFDataset
from palette_utils import posterize_image_to_palette, palette_indices_to_one_hot


class PixelArtDataSynthesizer:
    """Synthesizes training data for pixel art recovery."""
    
    def __init__(self, 
                 target_size: int,
                 input_size: int,
                 complexity_threshold: float = 10.0,
                 max_retries: int = 10,
                 transform_fraction: float = 0.0,
                 palette: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Args:
            target_size: Size of the target ground truth image (e.g., 32 for 32x32 target)
            input_size: Size of the final input image
            complexity_threshold: Minimum standard deviation for image complexity
            max_retries: Maximum attempts to find a complex enough crop
            transform_fraction: Fraction of samples to apply spatial transformations to (0.0-1.0)
            palette: Optional palette for posterization (N, 3) RGB values [0, 255]
            seed: Random seed for reproducibility
        """        
        # Internal crop size is 2x the target size to avoid black borders
        self.internal_crop_size = target_size * 2
        self.target_size = target_size
        self.input_size = input_size
        self.complexity_threshold = complexity_threshold
        self.max_retries = max_retries
        self.transform_fraction = transform_fraction
        self.palette = palette
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def random_crop(self, image: Image.Image) -> Image.Image:
        """Randomly crop a section from the input image (internal crop is 2x target size)."""
        width, height = image.size
        
        # Random crop coordinates
        left = random.randint(0, width - self.internal_crop_size)
        top = random.randint(0, height - self.internal_crop_size)
        
        return image.crop((left, top, left + self.internal_crop_size, top + self.internal_crop_size))
    
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
    
    def random_scale(self, image: Image.Image) -> Image.Image:
        """Scale image by random factor 2.0-8.0x with random interpolation."""
        # Random scale factors (can be different for x and y)
        base_scale = random.uniform(2.0, 8.0)
        scale_x = base_scale * random.uniform(0.9, 1.1)  # Allow slight variation
        scale_y = base_scale * random.uniform(0.9, 1.1)
        
        width, height = image.size
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        
        return image.resize((new_width, new_height), Image.NEAREST)
    
    def apply_jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Apply JPEG compression with quality between 70-95."""
        quality = random.randint(85, 99)
        
        # Save to bytes buffer with JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load back from buffer
        return Image.open(buffer).convert('RGB')
    
    def extract_center_region(self, image: Image.Image) -> Image.Image:
        """Extract the center target_size region from the internal 2x crop."""
        # The image is internal_crop_size x internal_crop_size
        # We want to extract target_size x target_size from the center
        center_x = center_y = self.internal_crop_size // 2
        half_target = self.target_size // 2
        left = center_x - half_target
        top = center_y - half_target
        right = left + self.target_size
        bottom = top + self.target_size
        return image.crop((left, top, right, bottom))
    
    def scale_to_final(self, image: Image.Image, transform_params: Tuple[float, float, float, float]) -> Image.Image:
        """Scale image to final size using precise coordinate mapping with optimized backends."""
        x_scale, y_scale, x_offset, y_offset = transform_params
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Generate coordinate grid for output
        output_size = self.input_size
        y_coords, x_coords = np.mgrid[0:output_size, 0:output_size]
        
        # Map output coordinates to input coordinates
        input_height, input_width = img_array.shape[:2]
        
        # Scale coordinates to map from [0, output_size-1] to [0, input_size-1]
        x_coords = x_coords * (input_width - 1) / (output_size - 1)
        y_coords = y_coords * (input_height - 1) / (output_size - 1)
        
        x_coords = (x_coords / 2) + ((input_width - 1)/4)
        y_coords = (y_coords / 2) + ((input_height - 1)/4)
        
        x_center = np.median(x_coords)
        x_coords = (x_coords - x_center) * np.exp(x_scale * 0.2) + x_center
        y_center = np.median(y_coords)
        y_coords = (y_coords - y_center) * np.exp(y_scale * 0.2) + y_center

        x_coords = x_coords + (x_offset / 64.0)
        y_coords = y_coords + (y_offset / 64.0)
        
        # Try OpenCV first (fastest), fallback to PyTorch, then scipy
        try:
            import cv2
            return self._scale_to_final_opencv(img_array, x_coords, y_coords, output_size)
        except ImportError:
            pass
            
        try:
            import torch
            return self._scale_to_final_pytorch(img_array, x_coords, y_coords, output_size)
        except ImportError:
            pass
        
        # Fallback to optimized scipy version
        return self._scale_to_final_scipy_optimized(img_array, x_coords, y_coords, output_size)
    
    def _scale_to_final_opencv(self, img_array, x_coords, y_coords, output_size):
        """OpenCV implementation - fastest option."""
        import cv2
        
        # OpenCV expects float32 maps
        map_x = x_coords.astype(np.float32)
        map_y = y_coords.astype(np.float32)
        
        # Use cv2.remap with cubic interpolation
        output_array = cv2.remap(
            img_array, 
            map_x, 
            map_y, 
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return Image.fromarray(output_array.astype(np.uint8))
    
    def _scale_to_final_pytorch(self, img_array, x_coords, y_coords, output_size):
        """PyTorch implementation - GPU accelerated if available."""
        import torch
        import torch.nn.functional as F
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to torch tensor and move to device
        if len(img_array.shape) == 3:
            # Convert from HWC to CHW format
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        else:
            img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Normalize coordinates to [-1, 1] range for grid_sample
        input_height, input_width = img_array.shape[:2]
        norm_x = 2.0 * x_coords / (input_width - 1) - 1.0
        norm_y = 2.0 * y_coords / (input_height - 1) - 1.0
        
        # Create grid for sampling
        grid = torch.stack([
            torch.from_numpy(norm_x).float(),
            torch.from_numpy(norm_y).float()
        ], dim=-1).unsqueeze(0).to(device)
        
        # Sample using bilinear interpolation
        output_tensor = F.grid_sample(
            img_tensor, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros',
            align_corners=True
        )
        
        # Convert back to numpy and PIL
        if len(img_array.shape) == 3:
            output_array = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            output_array = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
        
        return Image.fromarray(output_array.astype(np.uint8))
    
    def _scale_to_final_scipy_optimized(self, img_array, x_coords, y_coords, output_size):
        """Optimized scipy implementation with threading."""
        from scipy.ndimage import map_coordinates
        from concurrent.futures import ThreadPoolExecutor
        
        def process_channel(channel_data):
            return map_coordinates(
                channel_data,
                [y_coords, x_coords],
                order=3,
                mode='constant',
                cval=0
            )
        
        if len(img_array.shape) == 3:  # RGB image
            # Process channels in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_channel, img_array[:, :, c]) for c in range(3)]
                channels = [future.result() for future in futures]
            
            output_array = np.stack(channels, axis=-1)
        else:  # Grayscale
            output_array = process_channel(img_array)
        
        return Image.fromarray(output_array.astype(np.uint8))
    
    def generate_transform_params(self) -> Tuple[float, float, float, float]:
        """
        Generate random transformation parameters.
        
        Returns:
            Tuple of (x_scale, y_scale, x_offset, y_offset)
        """
        # Decide if this sample should be transformed
        if random.random() > self.transform_fraction:
            # No transformation (aligned sample)
            return 0.0, 0.0, 0.0, 0.0
        
        # Random shift in circular pattern (-1 to 1)
        angle = random.random() * 2 * math.pi
        distance = random.random()
        x_offset = distance * math.cos(angle)
        y_offset = distance * math.sin(angle)
        
        x_scale = y_scale = 0
        # Random scale (-1 to 1)
        if random.random() > 0.5:
            x_scale = random.random() * 2 - 1
        if random.random() > 0.5:
            y_scale = random.random() * 2 - 1
        
        return x_scale, y_scale, x_offset, y_offset
    
    def synthesize_pair(self, source_image: Image.Image) -> Tuple[Image.Image, Union[Image.Image, np.ndarray], Tuple[float, float, float, float]]:
        """
        Create a (input, ground_truth) pair from source image with optional spatial transformation.
        
        Returns:
            Tuple of (degraded_input, ground_truth_data, transform_params)
            - degraded_input: PIL Image for input
            - ground_truth_data: PIL Image (normal mode) or numpy array of indices (palette mode)
            - transform_params: Transformation parameters
        """
        # Step 1: Random crop with complexity checking
        for attempt in range(self.max_retries):
            cropped = self.random_crop(source_image)

            cropped = self.posterize_image(cropped)

            # Step 2: Extract center region (this becomes our ground truth)
            ground_truth = self.extract_center_region(cropped)
            
            # Check if the cropped region is too simple
            if not self.is_image_too_simple(ground_truth):
                break
        else:
            # If all attempts failed, use the last crop anyway
            # This prevents infinite loops with very simple source images
            pass
        
        
        # Step 3: Scale up with random interpolation
        scaled = self.random_scale(cropped)
        
        # Step 4: JPEG compression
        compressed = self.apply_jpeg_compression(scaled)
        
        # Generate transformation parameters first
        transform_params = self.generate_transform_params()
        
        # Step 5: Scale to final with transform parameters (this becomes our input)
        input_image = self.scale_to_final(compressed, transform_params)
        
        # Step 6: Handle palette mode if palette is provided
        if self.palette is not None:
            # Posterize ground truth to palette and return indices
            _, ground_truth_indices = posterize_image_to_palette(ground_truth, self.palette)
            return input_image, ground_truth_indices, transform_params
        else:
            # Return normal RGB ground truth
            return input_image, ground_truth, transform_params


class PixelArtDataset(Dataset):
    """PyTorch Dataset for pixel art recovery training."""
    
    def __init__(self, 
                 source_images_dir: str,
                 num_samples: int,
                 target_size: int,
                 input_size: int,
                 transform_fraction: float,
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
            transform_fraction: Fraction of samples to apply spatial transformations to (0.0-1.0)
            palette: Optional palette for posterization mode
        """
        self.source_images_dir = Path(source_images_dir)
        self.num_samples = num_samples
        self.synthesizer = synthesizer or PixelArtDataSynthesizer(
            target_size=target_size,
            input_size=input_size, 
            transform_fraction=transform_fraction,
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
        
        # Synthesize input/target pair with transformation parameters
        input_image, target_data, transform_params = self.synthesizer.synthesize_pair(source_image)
        
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
        
        # Create target data combining image and transformation parameters
        # For scale/offset training, we need both the image and transform params
        transform_tensor = torch.tensor(transform_params, dtype=torch.float32)
        
        # Return input tensor and combined target (image + transform params)
        return input_tensor, (target_tensor, transform_tensor)


class HuggingFacePixelArtDataset(Dataset):
    """PyTorch Dataset for pixel art recovery training using HuggingFace datasets."""
    
    def __init__(self, 
                 dataset_name: str,
                 num_samples: int,
                 target_size: int,
                 input_size: int,
                 transform_fraction: float,
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
            transform_fraction: Fraction of samples to apply spatial transformations to (0.0-1.0)
            palette: Optional palette for posterization mode
            **load_dataset_kwargs: Additional arguments to pass to load_dataset
        """        
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.synthesizer = synthesizer or PixelArtDataSynthesizer(
            target_size=target_size,
            input_size=input_size,
            transform_fraction=transform_fraction,
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
        
        # Synthesize input/target pair with transformation parameters
        input_image, target_data, transform_params = self.synthesizer.synthesize_pair(source_image)
        
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
        
        # Create target data combining image and transformation parameters
        # For scale/offset training, we need both the image and transform params
        transform_tensor = torch.tensor(transform_params, dtype=torch.float32)
        
        # Return input tensor and combined target (image + transform params)
        return input_tensor, (target_tensor, transform_tensor)


def create_hf_dataset(dataset_name: str,
                     num_samples: int,
                     target_size: int,
                     input_size: int,
                     save_examples: bool = True,
                     examples_dir: str = "dataset_examples",
                     split: str = "train",
                     image_column: str = "image",
                     streaming: bool = False,
                     transform_fraction: float = 0.0,
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
        transform_fraction=transform_fraction,
        palette=palette,
        **load_dataset_kwargs
    )
    
    if save_examples:
        os.makedirs(examples_dir, exist_ok=True)
        synthesizer = PixelArtDataSynthesizer(target_size=target_size, input_size=input_size, transform_fraction=transform_fraction, palette=palette, seed=69)  # Fixed seed for reproducible examples
        
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
            
            input_img, target_img, transform_params = synthesizer.synthesize_pair(source_image)
            
            # Save the pair
            input_img.save(f"{examples_dir}/hf_example_{i:02d}_input.png")
            target_img.save(f"{examples_dir}/hf_example_{i:02d}_target.png")
            
            # Save transformation info
            with open(f"{examples_dir}/hf_example_{i:02d}_transform.txt", "w") as f:
                f.write(f"Transform params: {transform_params}\n")
            
    
    return dataset


def create_dataset(source_images_dir: str, 
                  num_samples: int, 
                  target_size: int,
                  input_size: int,
                  save_examples: bool = False,
                  examples_dir: str = "dataset_examples",
                  transform_fraction: float = 0.0,
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
        transform_fraction=transform_fraction,
        palette=palette
    )
    
    if save_examples:
        os.makedirs(examples_dir, exist_ok=True)
        synthesizer = PixelArtDataSynthesizer(target_size=target_size, input_size=input_size, transform_fraction=transform_fraction, palette=palette, seed=42)  # Fixed seed for reproducible examples
        
        print(f"Saving example pairs to {examples_dir}/")
        for i in range(min(10, num_samples)):  # Limit examples to 10
            source_path = random.choice(dataset.source_image_paths)
            source_image = Image.open(source_path).convert('RGB')
            
            input_img, target_img, transform_params = synthesizer.synthesize_pair(source_image)
            
            # Save the pair
            input_img.save(f"{examples_dir}/example_{i:02d}_input.png")
            target_img.save(f"{examples_dir}/example_{i:02d}_target.png")
            
            # Save transformation info
            with open(f"{examples_dir}/example_{i:02d}_transform.txt", "w") as f:
                f.write(f"Transform params: {transform_params}\n")
    
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
    parser.add_argument("--transform_fraction", type=float, default=0.0,
                       help="Fraction of samples to apply spatial transformations to (0.0-1.0)")
    
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
            input_size=args.input_size,
            transform_fraction=args.transform_fraction
        )
    else:
        print(f"Creating dataset with {args.num_samples} samples from {args.source_dir}")
        dataset = create_dataset(
            source_images_dir=args.source_dir,
            num_samples=args.num_samples,
            save_examples=True,
            examples_dir=args.examples_dir,
            target_size=args.target_size,
            input_size=args.input_size,
            transform_fraction=args.transform_fraction
        )
    
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    # Test loading a sample
    input_tensor, target_data = dataset[0]
    target_tensor, transform_tensor = target_data
    print(f"Input shape: {input_tensor.shape}")   # Should be [3, input_size, input_size]
    print(f"Target shape: {target_tensor.shape}") # Should be [3, target_size, target_size]
    print(f"Transform shape: {transform_tensor.shape}") # Should be [4]
    print(f"Transform values: {transform_tensor}")  # Should show the transform parameters
