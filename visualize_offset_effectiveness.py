#!/usr/bin/env python3
"""
Script to visualize the effectiveness of the offset detection model.
Takes an image, applies various offset transformations, runs inference,
and creates a visualization of the results.
"""

import os
import sys
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import torch

# Add ml-training to path to import the inference module
sys.path.append(str(Path(__file__).parent / "ml-training"))
from infer_scale_offset import ScaleOffsetInference

from pixel_spacing_optimizer import rescale_image

class OffsetEffectivenessVisualizer:
    """Visualizes the effectiveness of offset detection across different input offsets."""
    
    def __init__(self, model_path: str, input_size: int = 128):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to the trained scale/offset detection model
            input_size: Input size for the model
        """
        self.model_path = model_path
        self.input_size = input_size
        
        # Initialize inference engine
        print("Loading scale/offset detection model...")
        self.inference = ScaleOffsetInference(
            model_path=model_path,
            input_size=input_size
        )
        print("Model loaded successfully!")
    
    def create_offset_images(self, 
                           original_image_path: str,
                           offset_range: Tuple[float, float] = (-0.5, 0.5),
                           num_steps: int = 9,
                           temp_dir: Optional[str] = None) -> List[Tuple[float, float, str]]:
        """
        Create offset versions of the input image.
        
        Args:
            original_image_path: Path to the original image
            offset_range: (min_offset, max_offset) for both axes
            num_steps: Number of offset steps in each direction
            temp_dir: Temporary directory for offset images
            
        Returns:
            List of (offset_x, offset_y, image_path) tuples
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="offset_test_")
        else:
            os.makedirs(temp_dir, exist_ok=True)
        
        print(f"Creating offset images in {temp_dir}")
        
        # Load original image
        original_image = Image.open(original_image_path).convert('RGB')
        original_width, original_height = original_image.size
        
        # Generate offset factors
        offsets = np.linspace(offset_range[0], offset_range[1], num_steps)
        
        offset_images = []
        
        for i, offset_x in enumerate(offsets):
            for j, offset_y in enumerate(offsets):
                # Apply offset transformation (no scaling, just offset)
                offset_image = rescale_image(original_image, 1.0, 1.0, offset_x, offset_y)
                
                # Save offset image
                offset_path = os.path.join(temp_dir, f"offset_{i:02d}_{j:02d}_ox{offset_x:.2f}_oy{offset_y:.2f}.png")
                offset_image.save(offset_path)
                
                offset_images.append((offset_x, offset_y, offset_path))
        
        print(f"Created {len(offset_images)} offset images")
        return offset_images

    
    def run_inference_on_offset_images(self, 
                                     offset_images: List[Tuple[float, float, str]]) -> List[Dict]:
        """
        Run offset inference on all offset images.
        
        Args:
            offset_images: List of (offset_x, offset_y, image_path) tuples
            
        Returns:
            List of inference results with ground truth offsets
        """
        print("Running inference on offset images...")
        
        results = []
        
        for true_offset_x, true_offset_y, image_path in offset_images:
            try:
                # Run inference
                result = self.inference.predict_single(image_path)
                
                # Add ground truth information
                result['true_offset_x'] = true_offset_x
                result['true_offset_y'] = true_offset_y
                result['image_path'] = image_path
                
                # Calculate errors
                result['error_offset_x'] = result['offset_x'] - true_offset_x
                result['error_offset_y'] = result['offset_y'] - true_offset_y
                result['abs_error_offset_x'] = abs(result['error_offset_x'])
                result['abs_error_offset_y'] = abs(result['error_offset_y'])
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'true_offset_x': true_offset_x,
                    'true_offset_y': true_offset_y,
                    'image_path': image_path,
                    'error': str(e)
                })
        
        valid_results = [r for r in results if 'error' not in r]
        print(f"Successfully processed {len(valid_results)}/{len(results)} images")
        
        return results
    
    def create_visualization(self, 
                           results: List[Dict],
                           output_path: str,
                           original_image_path: str):
        """
        Create a comprehensive visualization of the offset detection effectiveness.
        
        Args:
            results: Inference results
            output_path: Path to save the visualization
            original_image_path: Path to the original image
        """
        # Filter valid results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        print(f"Creating visualization with {len(valid_results)} data points...")
        
        # Extract data for visualization
        true_offsets_x = np.array([r['true_offset_x'] for r in valid_results])
        true_offsets_y = np.array([r['true_offset_y'] for r in valid_results])
        pred_offsets_x = np.array([r['offset_x'] for r in valid_results])
        pred_offsets_y = np.array([r['offset_y'] for r in valid_results])
        errors_x = np.array([r['error_offset_x'] for r in valid_results])
        errors_y = np.array([r['error_offset_y'] for r in valid_results])
        
        # Get unique offset values for grid
        unique_offsets_x = np.unique(true_offsets_x)
        unique_offsets_y = np.unique(true_offsets_y)
        
        # Create grid for heatmaps
        grid_shape = (len(unique_offsets_y), len(unique_offsets_x))
        pred_x_grid = np.full(grid_shape, np.nan)
        pred_y_grid = np.full(grid_shape, np.nan)
        error_x_grid = np.full(grid_shape, np.nan)
        error_y_grid = np.full(grid_shape, np.nan)
        
        # Fill grids
        for result in valid_results:
            i = np.where(unique_offsets_y == result['true_offset_y'])[0][0]
            j = np.where(unique_offsets_x == result['true_offset_x'])[0][0]
            
            pred_x_grid[i, j] = result['offset_x']
            pred_y_grid[i, j] = result['offset_y']
            error_x_grid[i, j] = result['error_offset_x']
            error_y_grid[i, j] = result['error_offset_y']
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # Original image
        ax1 = plt.subplot(3, 4, 1)
        original_img = Image.open(original_image_path).convert('RGB')
        ax1.imshow(original_img)
        ax1.set_title(f"Original Image\n{Path(original_image_path).name}", fontsize=12)
        ax1.axis('off')
        
        # Predicted Offset X heatmap
        ax2 = plt.subplot(3, 4, 2)
        pred_x_max = max(abs(np.nanmin(pred_x_grid)), abs(np.nanmax(pred_x_grid)))
        im2 = ax2.imshow(pred_x_grid, aspect='auto', origin='lower', 
                        extent=[unique_offsets_x.min(), unique_offsets_x.max(),
                               unique_offsets_y.min(), unique_offsets_y.max()],
                        cmap='RdBu_r', vmin=-pred_x_max, vmax=pred_x_max)
        ax2.set_title('Predicted Offset X', fontsize=12)
        ax2.set_xlabel('True Offset X')
        ax2.set_ylabel('True Offset Y')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Predicted Offset Y heatmap
        ax3 = plt.subplot(3, 4, 3)
        pred_y_max = max(abs(np.nanmin(pred_y_grid)), abs(np.nanmax(pred_y_grid)))
        im3 = ax3.imshow(pred_y_grid, aspect='auto', origin='lower',
                        extent=[unique_offsets_x.min(), unique_offsets_x.max(),
                               unique_offsets_y.min(), unique_offsets_y.max()],
                        cmap='RdBu_r', vmin=-pred_y_max, vmax=pred_y_max)
        ax3.set_title('Predicted Offset Y', fontsize=12)
        ax3.set_xlabel('True Offset X')
        ax3.set_ylabel('True Offset Y')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # Error heatmaps
        ax4 = plt.subplot(3, 4, 4)
        max_error = max(abs(errors_x.min()), abs(errors_x.max()))
        im4 = ax4.imshow(error_x_grid, aspect='auto', origin='lower',
                        extent=[unique_offsets_x.min(), unique_offsets_x.max(),
                               unique_offsets_y.min(), unique_offsets_y.max()],
                        cmap='RdBu_r', vmin=-max_error, vmax=max_error)
        ax4.set_title('Offset X Error\n(Predicted - True)', fontsize=12)
        ax4.set_xlabel('True Offset X')
        ax4.set_ylabel('True Offset Y')
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        ax5 = plt.subplot(3, 4, 5)
        max_error_y = max(abs(errors_y.min()), abs(errors_y.max()))
        im5 = ax5.imshow(error_y_grid, aspect='auto', origin='lower',
                        extent=[unique_offsets_x.min(), unique_offsets_x.max(),
                               unique_offsets_y.min(), unique_offsets_y.max()],
                        cmap='RdBu_r', vmin=-max_error_y, vmax=max_error_y)
        ax5.set_title('Offset Y Error\n(Predicted - True)', fontsize=12)
        ax5.set_xlabel('True Offset X')
        ax5.set_ylabel('True Offset Y')
        plt.colorbar(im5, ax=ax5, shrink=0.8)
        
        # Scatter plots: Predicted vs True
        ax6 = plt.subplot(3, 4, 6)
        ax6.scatter(true_offsets_x, pred_offsets_x, alpha=0.7, c='blue', s=50)
        min_offset = min(true_offsets_x.min(), pred_offsets_x.min())
        max_offset = max(true_offsets_x.max(), pred_offsets_x.max())
        ax6.plot([min_offset, max_offset], [min_offset, max_offset], 'r--', alpha=0.8)
        ax6.set_xlabel('True Offset X')
        ax6.set_ylabel('Predicted Offset X')
        ax6.set_title('Offset X: Predicted vs True')
        ax6.grid(True, alpha=0.3)
        
        ax7 = plt.subplot(3, 4, 7)
        ax7.scatter(true_offsets_y, pred_offsets_y, alpha=0.7, c='green', s=50)
        min_offset = min(true_offsets_y.min(), pred_offsets_y.min())
        max_offset = max(true_offsets_y.max(), pred_offsets_y.max())
        ax7.plot([min_offset, max_offset], [min_offset, max_offset], 'r--', alpha=0.8)
        ax7.set_xlabel('True Offset Y')
        ax7.set_ylabel('Predicted Offset Y')
        ax7.set_title('Offset Y: Predicted vs True')
        ax7.grid(True, alpha=0.3)
        
        # Error distribution histograms
        ax8 = plt.subplot(3, 4, 8)
        ax8.hist(errors_x, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax8.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax8.set_xlabel('Offset X Error')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Offset X Error Distribution')
        ax8.grid(True, alpha=0.3)
        
        ax9 = plt.subplot(3, 4, 9)
        ax9.hist(errors_y, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax9.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax9.set_xlabel('Offset Y Error')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Offset Y Error Distribution')
        ax9.grid(True, alpha=0.3)
        
        # Statistics text
        ax10 = plt.subplot(3, 4, 10)
        stats_text = f"""Offset Detection Statistics:

Offset X:
  Mean Error: {errors_x.mean():.4f} ± {errors_x.std():.4f}
  MAE: {np.abs(errors_x).mean():.4f}
  RMSE: {np.sqrt((errors_x**2).mean()):.4f}
  R²: {np.corrcoef(true_offsets_x, pred_offsets_x)[0,1]**2:.4f}

Offset Y:
  Mean Error: {errors_y.mean():.4f} ± {errors_y.std():.4f}
  MAE: {np.abs(errors_y).mean():.4f}
  RMSE: {np.sqrt((errors_y**2).mean()):.4f}
  R²: {np.corrcoef(true_offsets_y, pred_offsets_y)[0,1]**2:.4f}

Dataset:
  Offset Range: [{unique_offsets_x.min():.1f}, {unique_offsets_x.max():.1f}]
  Grid Size: {len(unique_offsets_x)} × {len(unique_offsets_y)}
  Valid Samples: {len(valid_results)}/{len(results)}"""
        
        ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, 
                 fontsize=10, family='monospace', verticalalignment='top')
        ax10.set_xlim(0, 1)
        ax10.set_ylim(0, 1)
        ax10.axis('off')
        
        # Error vs offset magnitude
        ax11 = plt.subplot(3, 4, 11)
        offset_magnitude = np.sqrt(true_offsets_x**2 + true_offsets_y**2)
        error_magnitude = np.sqrt(errors_x**2 + errors_y**2)
        ax11.scatter(offset_magnitude, error_magnitude, alpha=0.7, s=50)
        ax11.set_xlabel('True Offset Magnitude')
        ax11.set_ylabel('Error Magnitude')
        ax11.set_title('Error vs Offset Magnitude')
        ax11.grid(True, alpha=0.3)
        
        # 2D error visualization
        ax12 = plt.subplot(3, 4, 12)
        ax12.quiver(true_offsets_x, true_offsets_y, errors_x, errors_y, 
                   angles='xy', scale_units='xy', scale=1, alpha=0.7, width=0.003)
        ax12.set_xlabel('True Offset X')
        ax12.set_ylabel('True Offset Y')
        ax12.set_title('Error Vectors\n(Arrow = Prediction Error)')
        ax12.grid(True, alpha=0.3)
        ax12.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def run_analysis(self, 
                    image_path: str,
                    output_path: str,
                    offset_range: Tuple[float, float] = (-0.5, 0.5),
                    num_steps: int = 9,
                    temp_dir: Optional[str] = None,
                    cleanup: bool = True):
        """
        Run the complete offset effectiveness analysis.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            offset_range: (min_offset, max_offset) for both axes  
            num_steps: Number of offset steps in each direction
            temp_dir: Temporary directory for offset images
            cleanup: Whether to clean up temporary files
        """
        print(f"Starting offset effectiveness analysis for {image_path}")
        print(f"Offset range: {offset_range}, Steps: {num_steps}")
        
        # Create offset images
        offset_images = self.create_offset_images(
            original_image_path=image_path,
            offset_range=offset_range,
            num_steps=num_steps,
            temp_dir=temp_dir
        )
        
        # Run inference
        results = self.run_inference_on_offset_images(offset_images)
        
        # Create visualization
        self.create_visualization(
            results=results,
            output_path=output_path,
            original_image_path=image_path
        )
        
        # Cleanup temporary files
        if cleanup and temp_dir:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        
        # Print summary
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            errors_x = [r['error_offset_x'] for r in valid_results]
            errors_y = [r['error_offset_y'] for r in valid_results]
            
            print(f"\n=== Analysis Complete ===")
            print(f"Processed {len(valid_results)}/{len(results)} images successfully")
            print(f"Offset X - MAE: {np.mean(np.abs(errors_x)):.4f}, RMSE: {np.sqrt(np.mean([e**2 for e in errors_x])):.4f}")
            print(f"Offset Y - MAE: {np.mean(np.abs(errors_y)):.4f}, RMSE: {np.sqrt(np.mean([e**2 for e in errors_y])):.4f}")
            print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize offset detection model effectiveness")
    
    parser.add_argument("--image", type=str, required=True,
                       help="Input image path")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained scale/offset detection model")
    parser.add_argument("--output", type=str, default="offset_effectiveness_visualization.png",
                       help="Output visualization path")
    
    # Offset testing parameters
    parser.add_argument("--offset_min", type=float, default=-0.5,
                       help="Minimum offset factor")
    parser.add_argument("--offset_max", type=float, default=0.5,
                       help="Maximum offset factor")
    parser.add_argument("--num_steps", type=int, default=9,
                       help="Number of offset steps in each direction")
    
    # System parameters
    parser.add_argument("--input_size", type=int, default=128,
                       help="Model input size")
    parser.add_argument("--temp_dir", type=str,
                       help="Temporary directory for offset images")
    parser.add_argument("--keep_temp", action="store_true",
                       help="Keep temporary files after processing")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    try:
        # Create visualizer
        visualizer = OffsetEffectivenessVisualizer(
            model_path=args.model_path,
            input_size=args.input_size
        )
        
        # Run analysis
        visualizer.run_analysis(
            image_path=args.image,
            output_path=args.output,
            offset_range=(args.offset_min, args.offset_max),
            num_steps=args.num_steps,
            temp_dir=args.temp_dir,
            cleanup=not args.keep_temp
        )
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())