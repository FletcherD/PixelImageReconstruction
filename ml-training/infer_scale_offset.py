#!/usr/bin/env python3
"""
Inference script for scale/offset detection model.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from scale_offset_model import ScaleOffsetDetector


class ScaleOffsetInference:
    """Inference engine for scale/offset detection."""
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[torch.device] = None,
                 input_size: int = 128):
        """
        Args:
            model_path: Path to the trained model checkpoint
            device: Inference device
            input_size: Input image size
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = ScaleOffsetDetector(input_size=input_size)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path or PIL Image")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict_single(self, image: Union[str, Path, Image.Image]) -> dict:
        """
        Predict scale and offset for a single image.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            scale, offset = self.model.predict_transform_params(input_tensor)
        
        # Convert to numpy
        output_np = output.cpu().numpy()[0]  # Remove batch dimension
        scale_np = scale.cpu().numpy()[0]
        offset_np = offset.cpu().numpy()[0]
        
        return {
            'raw_output': output_np,
            'scale_x': float(scale_np[0]),
            'scale_y': float(scale_np[1]),
            'offset_x': float(offset_np[0]),
            'offset_y': float(offset_np[1]),
            'scale': scale_np,
            'offset': offset_np
        }
    
    def predict_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[dict]:
        """
        Predict scale and offset for a batch of images.
        
        Args:
            images: List of image paths or PIL Images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image in tqdm(images, desc="Processing images"):
            try:
                result = self.predict_single(image)
                result['image'] = str(image) if isinstance(image, (str, Path)) else "PIL_Image"
                results.append(result)
            except Exception as e:
                print(f"Error processing {image}: {e}")
                results.append({
                    'image': str(image) if isinstance(image, (str, Path)) else "PIL_Image",
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, 
                         input_dir: str, 
                         output_file: Optional[str] = None,
                         image_extensions: List[str] = None) -> List[dict]:
        """
        Predict scale and offset for all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_file: Optional output file to save results
            image_extensions: List of image extensions to process
            
        Returns:
            List of prediction dictionaries
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        input_path = Path(input_dir)
        image_files = [
            p for p in input_path.rglob('*')
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return []
        
        print(f"Found {len(image_files)} images")
        
        # Process images
        results = self.predict_batch(image_files)
        
        # Save results if requested
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[dict], output_file: str):
        """Save prediction results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def visualize_predictions(self, 
                            results: List[dict], 
                            output_dir: str,
                            max_images: int = 20):
        """
        Create visualizations of predictions.
        
        Args:
            results: Prediction results
            output_dir: Output directory for visualizations
            max_images: Maximum number of images to visualize
        """
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Limit number of images
        valid_results = valid_results[:max_images]
        
        print(f"Creating visualizations for {len(valid_results)} images")
        
        for i, result in enumerate(tqdm(valid_results, desc="Creating visualizations")):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Load and display original image
            if 'image' in result and result['image'] != "PIL_Image":
                try:
                    image = Image.open(result['image']).convert('RGB')
                    ax1.imshow(image)
                    ax1.set_title(f"Original Image\n{Path(result['image']).name}")
                except:
                    ax1.text(0.5, 0.5, "Image not available", ha='center', va='center')
                    ax1.set_title("Original Image")
            else:
                ax1.text(0.5, 0.5, "PIL Image", ha='center', va='center')
                ax1.set_title("Original Image")
            
            ax1.axis('off')
            
            # Display predictions as text
            pred_text = f"Predicted Transformation:\n\n"
            pred_text += f"Scale X: {result['scale_x']:.3f}\n"
            pred_text += f"Scale Y: {result['scale_y']:.3f}\n"
            pred_text += f"Offset X: {result['offset_x']:.3f}\n"
            pred_text += f"Offset Y: {result['offset_y']:.3f}\n\n"
            pred_text += f"Raw Output:\n{result['raw_output']}"
            
            ax2.text(0.1, 0.9, pred_text, fontsize=10, family='monospace',
                    verticalalignment='top', transform=ax2.transAxes)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title("Predictions")
            
            plt.tight_layout()
            plt.savefig(output_path / f"prediction_{i:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Inference for scale/offset detection model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--input_size", type=int, default=128,
                       help="Input image size")
    
    # Input arguments
    parser.add_argument("--image", type=str,
                       help="Single image to process")
    parser.add_argument("--input_dir", type=str,
                       help="Directory containing images to process")
    parser.add_argument("--image_list", type=str,
                       help="Text file containing list of image paths")
    
    # Output arguments
    parser.add_argument("--output_file", type=str,
                       help="Output file to save results (JSON)")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for visualizations")
    parser.add_argument("--max_visualizations", type=int, default=20,
                       help="Maximum number of visualizations to create")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Inference device (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Check that at least one input is provided
    if not any([args.image, args.input_dir, args.image_list]):
        parser.error("Provide at least one of --image, --input_dir, or --image_list")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Create inference engine
    inference = ScaleOffsetInference(
        model_path=args.model_path,
        device=device,
        input_size=args.input_size
    )
    
    results = []
    
    # Process single image
    if args.image:
        print(f"Processing single image: {args.image}")
        result = inference.predict_single(args.image)
        result['image'] = args.image
        results.append(result)
        
        # Print results
        print(f"\nPrediction for {args.image}:")
        print(f"Scale X: {result['scale_x']:.4f}")
        print(f"Scale Y: {result['scale_y']:.4f}")
        print(f"Offset X: {result['offset_x']:.4f}")
        print(f"Offset Y: {result['offset_y']:.4f}")
    
    # Process directory
    if args.input_dir:
        print(f"Processing directory: {args.input_dir}")
        dir_results = inference.predict_directory(
            input_dir=args.input_dir,
            output_file=args.output_file
        )
        results.extend(dir_results)
    
    # Process image list
    if args.image_list:
        print(f"Processing image list: {args.image_list}")
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        list_results = inference.predict_batch(image_paths)
        results.extend(list_results)
    
    # Save results
    if args.output_file and results:
        inference.save_results(results, args.output_file)
    
    # Create visualizations
    if args.output_dir and results:
        inference.visualize_predictions(
            results=results,
            output_dir=args.output_dir,
            max_images=args.max_visualizations
        )
    
    # Print summary
    if results:
        valid_results = [r for r in results if 'error' not in r]
        print(f"\nProcessed {len(results)} images ({len(valid_results)} successful)")
        
        if valid_results:
            scales_x = [r['scale_x'] for r in valid_results]
            scales_y = [r['scale_y'] for r in valid_results]
            offsets_x = [r['offset_x'] for r in valid_results]
            offsets_y = [r['offset_y'] for r in valid_results]
            
            print(f"\nSummary Statistics:")
            print(f"Scale X - Mean: {np.mean(scales_x):.3f}, Std: {np.std(scales_x):.3f}")
            print(f"Scale Y - Mean: {np.mean(scales_y):.3f}, Std: {np.std(scales_y):.3f}")
            print(f"Offset X - Mean: {np.mean(offsets_x):.3f}, Std: {np.std(offsets_x):.3f}")
            print(f"Offset Y - Mean: {np.mean(offsets_y):.3f}, Std: {np.std(offsets_y):.3f}")


if __name__ == "__main__":
    main()