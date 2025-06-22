#!/usr/bin/env python3
"""
Base class and utilities for effectiveness visualization scripts.
Contains common code shared between scale and offset effectiveness visualizers.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Add ml-training to path to import the inference module
sys.path.append(str(Path(__file__).parent / "ml-training"))
from infer_scale_offset import ScaleOffsetInference

from pixel_spacing_optimizer import rescale_image


class EffectivenessVisualizerBase(ABC):
    """Base class for effectiveness visualizers."""
    
    def __init__(self, model_path: str, input_size: int = 128):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to the trained model
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
    
    @abstractmethod
    def create_transformed_images(self, 
                                original_image_path: str,
                                param_range: Tuple[float, float],
                                num_steps: int,
                                temp_dir: Optional[str] = None) -> List[Tuple[float, float, str]]:
        """
        Create transformed versions of the input image.
        Must be implemented by subclasses.
        """
        pass
    
    def run_inference_on_images(self, 
                              transformed_images: List[Tuple[float, float, str]],
                              param_name: str,
                              batch_size: int = 32) -> List[Dict]:
        """
        Run inference on all transformed images using batch processing.
        
        Args:
            transformed_images: List of (param_x, param_y, image_path) tuples
            param_name: Parameter name (e.g., 'scale', 'offset')
            batch_size: Batch size for inference
            
        Returns:
            List of inference results with ground truth parameters
        """
        print(f"Running batch inference on {len(transformed_images)} images...")
        
        results = []
        
        # Process in batches
        for i in range(0, len(transformed_images), batch_size):
            batch_images = transformed_images[i:i + batch_size]
            batch_paths = [img_path for _, _, img_path in batch_images]
            batch_params = [(param_x, param_y) for param_x, param_y, _ in batch_images]
            
            try:
                # Run individual inference with patches (for accuracy matching original behavior)
                batch_results = []
                for img_path in batch_paths:
                    result = self.inference.predict_single(img_path, use_patches=True)
                    batch_results.append(result)
                
                # Add ground truth information and calculate errors
                for j, result in enumerate(batch_results):
                    true_param_x, true_param_y = batch_params[j]
                    
                    # Add ground truth information
                    result[f'true_{param_name}_x'] = true_param_x
                    result[f'true_{param_name}_y'] = true_param_y
                    result['image_path'] = batch_paths[j]
                    
                    # Calculate errors
                    result[f'error_{param_name}_x'] = result[f'{param_name}_x'] - true_param_x
                    result[f'error_{param_name}_y'] = result[f'{param_name}_y'] - true_param_y
                    result[f'abs_error_{param_name}_x'] = abs(result[f'error_{param_name}_x'])
                    result[f'abs_error_{param_name}_y'] = abs(result[f'error_{param_name}_y'])
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add error entries for failed batch
                for j, (true_param_x, true_param_y, img_path) in enumerate(batch_images):
                    results.append({
                        f'true_{param_name}_x': true_param_x,
                        f'true_{param_name}_y': true_param_y,
                        'image_path': img_path,
                        'error': str(e)
                    })
        
        valid_results = [r for r in results if 'error' not in r]
        print(f"Successfully processed {len(valid_results)}/{len(results)} images")
        
        return results
    
    def create_comprehensive_visualization(self, 
                                        results: List[Dict],
                                        output_path: str,
                                        original_image_path: str,
                                        param_name: str,
                                        param_display_name: str):
        """
        Create a comprehensive visualization of the detection effectiveness.
        
        Args:
            results: Inference results
            output_path: Path to save the visualization
            original_image_path: Path to the original image
            param_name: Parameter name (e.g., 'scale', 'offset')
            param_display_name: Display name for parameter (e.g., 'Scale', 'Offset')
        """
        # Filter valid results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        print(f"Creating visualization with {len(valid_results)} data points...")
        
        # Extract data for visualization
        true_params_x = np.array([r[f'true_{param_name}_x'] for r in valid_results])
        true_params_y = np.array([r[f'true_{param_name}_y'] for r in valid_results])
        pred_params_x = np.array([r[f'{param_name}_x'] for r in valid_results])
        pred_params_y = np.array([r[f'{param_name}_y'] for r in valid_results])
        errors_x = np.array([r[f'error_{param_name}_x'] for r in valid_results])
        errors_y = np.array([r[f'error_{param_name}_y'] for r in valid_results])
        variances_x = np.array([r.get(f'{param_name}_x_std', 0.0) for r in valid_results])
        variances_y = np.array([r.get(f'{param_name}_y_std', 0.0) for r in valid_results])
        
        # Get unique parameter values for grid
        unique_params_x = np.unique(true_params_x)
        unique_params_y = np.unique(true_params_y)
        
        # Create grid for heatmaps
        grid_shape = (len(unique_params_y), len(unique_params_x))
        pred_x_grid = np.full(grid_shape, np.nan)
        pred_y_grid = np.full(grid_shape, np.nan)
        variance_x_grid = np.full(grid_shape, np.nan)
        variance_y_grid = np.full(grid_shape, np.nan)
        
        # Fill grids
        for result in valid_results:
            i = np.where(unique_params_y == result[f'true_{param_name}_y'])[0][0]
            j = np.where(unique_params_x == result[f'true_{param_name}_x'])[0][0]
            
            pred_x_grid[i, j] = result[f'{param_name}_x']
            pred_y_grid[i, j] = result[f'{param_name}_y']
            variance_x_grid[i, j] = result.get(f'{param_name}_x_std', 0.0)
            variance_y_grid[i, j] = result.get(f'{param_name}_y_std', 0.0)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # Original image
        ax1 = plt.subplot(3, 4, 1)
        original_img = Image.open(original_image_path).convert('RGB')
        ax1.imshow(original_img)
        ax1.set_title(f"Original Image\n{Path(original_image_path).name}", fontsize=12)
        ax1.axis('off')
        
        # Predicted Parameter X heatmap
        ax2 = plt.subplot(3, 4, 2)
        pred_x_max = max(abs(np.nanmin(pred_x_grid)), abs(np.nanmax(pred_x_grid)))
        if pred_x_max > 0:
            im2 = ax2.imshow(pred_x_grid, aspect='auto', origin='lower', 
                            extent=[unique_params_x.min(), unique_params_x.max(),
                                   unique_params_y.min(), unique_params_y.max()],
                            cmap='RdBu_r', vmin=-pred_x_max, vmax=pred_x_max)
            plt.colorbar(im2, ax=ax2, shrink=0.8)
        ax2.set_title(f'Predicted {param_display_name} X', fontsize=12)
        ax2.set_xlabel(f'True {param_display_name} X')
        ax2.set_ylabel(f'True {param_display_name} Y')
        
        # Predicted Parameter Y heatmap
        ax3 = plt.subplot(3, 4, 3)
        pred_y_max = max(abs(np.nanmin(pred_y_grid)), abs(np.nanmax(pred_y_grid)))
        if pred_y_max > 0:
            im3 = ax3.imshow(pred_y_grid, aspect='auto', origin='lower',
                            extent=[unique_params_x.min(), unique_params_x.max(),
                                   unique_params_y.min(), unique_params_y.max()],
                            cmap='RdBu_r', vmin=-pred_y_max, vmax=pred_y_max)
            plt.colorbar(im3, ax=ax3, shrink=0.8)
        ax3.set_title(f'Predicted {param_display_name} Y', fontsize=12)
        ax3.set_xlabel(f'True {param_display_name} X')
        ax3.set_ylabel(f'True {param_display_name} Y')
        
        # Variance heatmaps
        ax4 = plt.subplot(3, 4, 4)
        max_var_x = np.nanmax(variance_x_grid) if not np.all(np.isnan(variance_x_grid)) else 1
        if max_var_x > 0:
            im4 = ax4.imshow(variance_x_grid, aspect='auto', origin='lower',
                            extent=[unique_params_x.min(), unique_params_x.max(),
                                   unique_params_y.min(), unique_params_y.max()],
                            cmap='viridis', vmin=0, vmax=max_var_x)
            plt.colorbar(im4, ax=ax4, shrink=0.8)
        ax4.set_title(f'{param_display_name} X Variance\n(Across Patches)', fontsize=12)
        ax4.set_xlabel(f'True {param_display_name} X')
        ax4.set_ylabel(f'True {param_display_name} Y')
        
        ax5 = plt.subplot(3, 4, 5)
        max_var_y = np.nanmax(variance_y_grid) if not np.all(np.isnan(variance_y_grid)) else 1
        if max_var_y > 0:
            im5 = ax5.imshow(variance_y_grid, aspect='auto', origin='lower',
                            extent=[unique_params_x.min(), unique_params_x.max(),
                                   unique_params_y.min(), unique_params_y.max()],
                            cmap='viridis', vmin=0, vmax=max_var_y)
            plt.colorbar(im5, ax=ax5, shrink=0.8)
        ax5.set_title(f'{param_display_name} Y Variance\n(Across Patches)', fontsize=12)
        ax5.set_xlabel(f'True {param_display_name} X')
        ax5.set_ylabel(f'True {param_display_name} Y')
        
        # Scatter plots: Predicted vs True
        ax6 = plt.subplot(3, 4, 6)
        ax6.scatter(true_params_x, pred_params_x, alpha=0.7, c='blue', s=50)
        min_param = min(true_params_x.min(), pred_params_x.min())
        max_param = max(true_params_x.max(), pred_params_x.max())
        ax6.plot([min_param, max_param], [min_param, max_param], 'r--', alpha=0.8)
        ax6.set_xlabel(f'True {param_display_name} X')
        ax6.set_ylabel(f'Predicted {param_display_name} X')
        ax6.set_title(f'{param_display_name} X: Predicted vs True')
        ax6.grid(True, alpha=0.3)
        
        ax7 = plt.subplot(3, 4, 7)
        ax7.scatter(true_params_y, pred_params_y, alpha=0.7, c='green', s=50)
        min_param = min(true_params_y.min(), pred_params_y.min())
        max_param = max(true_params_y.max(), pred_params_y.max())
        ax7.plot([min_param, max_param], [min_param, max_param], 'r--', alpha=0.8)
        ax7.set_xlabel(f'True {param_display_name} Y')
        ax7.set_ylabel(f'Predicted {param_display_name} Y')
        ax7.set_title(f'{param_display_name} Y: Predicted vs True')
        ax7.grid(True, alpha=0.3)
        
        # Variance distribution histograms
        ax8 = plt.subplot(3, 4, 8)
        ax8.hist(variances_x, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax8.axvline(np.mean(variances_x), color='red', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(variances_x):.3f}')
        ax8.set_xlabel(f'{param_display_name} X Variance')
        ax8.set_ylabel('Frequency')
        ax8.set_title(f'{param_display_name} X Variance Distribution')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        ax9 = plt.subplot(3, 4, 9)
        ax9.hist(variances_y, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax9.axvline(np.mean(variances_y), color='red', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(variances_y):.3f}')
        ax9.set_xlabel(f'{param_display_name} Y Variance')
        ax9.set_ylabel('Frequency')
        ax9.set_title(f'{param_display_name} Y Variance Distribution')
        ax9.grid(True, alpha=0.3)
        ax9.legend()
        
        # Statistics text
        ax10 = plt.subplot(3, 4, 10)
        r2_x = np.corrcoef(true_params_x, pred_params_x)[0,1]**2 if len(true_params_x) > 1 else 0
        r2_y = np.corrcoef(true_params_y, pred_params_y)[0,1]**2 if len(true_params_y) > 1 else 0
        
        stats_text = f"""{param_display_name} Detection Statistics:

{param_display_name} X:
  Mean Error: {errors_x.mean():.4f} ± {errors_x.std():.4f}
  MAE: {np.abs(errors_x).mean():.4f}
  RMSE: {np.sqrt((errors_x**2).mean()):.4f}
  R²: {r2_x:.4f}
  Mean Variance: {variances_x.mean():.4f}

{param_display_name} Y:
  Mean Error: {errors_y.mean():.4f} ± {errors_y.std():.4f}
  MAE: {np.abs(errors_y).mean():.4f}
  RMSE: {np.sqrt((errors_y**2).mean()):.4f}
  R²: {r2_y:.4f}
  Mean Variance: {variances_y.mean():.4f}

Dataset:
  {param_display_name} Range: [{unique_params_x.min():.1f}, {unique_params_x.max():.1f}]
  Grid Size: {len(unique_params_x)} × {len(unique_params_y)}
  Valid Samples: {len(valid_results)}/{len(results)}"""
        
        ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, 
                 fontsize=10, family='monospace', verticalalignment='top')
        ax10.set_xlim(0, 1)
        ax10.set_ylim(0, 1)
        ax10.axis('off')
        
        # Variance vs parameter magnitude
        ax11 = plt.subplot(3, 4, 11)
        param_magnitude = np.sqrt(true_params_x**2 + true_params_y**2)
        variance_magnitude = np.sqrt(variances_x**2 + variances_y**2)
        ax11.scatter(param_magnitude, variance_magnitude, alpha=0.7, s=50)
        ax11.set_xlabel(f'True {param_display_name} Magnitude')
        ax11.set_ylabel('Variance Magnitude')
        ax11.set_title(f'Variance vs {param_display_name} Magnitude')
        ax11.grid(True, alpha=0.3)
        
        # 2D variance visualization
        ax12 = plt.subplot(3, 4, 12)
        # Create scatter plot with variance as color and size
        variance_total = variances_x + variances_y
        scatter = ax12.scatter(true_params_x, true_params_y, 
                              c=variance_total, s=50 + variance_total*200, 
                              alpha=0.7, cmap='plasma')
        ax12.set_xlabel(f'True {param_display_name} X')
        ax12.set_ylabel(f'True {param_display_name} Y')
        ax12.set_title('Variance Map\n(Color & Size = Total Variance)')
        ax12.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax12, shrink=0.8, label='Total Variance')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def run_analysis(self, 
                    image_path: str,
                    output_path: str,
                    param_range: Tuple[float, float],
                    num_steps: int = 9,
                    temp_dir: Optional[str] = None,
                    cleanup: bool = True,
                    param_name: str = "param",
                    param_display_name: str = "Parameter",
                    batch_size: int = 32):
        """
        Run the complete effectiveness analysis.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            param_range: (min_param, max_param) for both axes  
            num_steps: Number of parameter steps in each direction
            temp_dir: Temporary directory for transformed images
            cleanup: Whether to clean up temporary files
            param_name: Parameter name for internal use
            param_display_name: Parameter display name for visualization
        """
        print(f"Starting {param_name} effectiveness analysis for {image_path}")
        print(f"{param_display_name} range: {param_range}, Steps: {num_steps}")
        
        # Create transformed images
        transformed_images = self.create_transformed_images(
            original_image_path=image_path,
            param_range=param_range,
            num_steps=num_steps,
            temp_dir=temp_dir
        )
        
        # Run inference
        results = self.run_inference_on_images(transformed_images, param_name, batch_size)
        
        # Create visualization
        self.create_comprehensive_visualization(
            results=results,
            output_path=output_path,
            original_image_path=image_path,
            param_name=param_name,
            param_display_name=param_display_name
        )
        
        # Cleanup temporary files
        if cleanup and temp_dir:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        
        # Print summary
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            errors_x = [r[f'error_{param_name}_x'] for r in valid_results]
            errors_y = [r[f'error_{param_name}_y'] for r in valid_results]
            
            print(f"\n=== Analysis Complete ===")
            print(f"Processed {len(valid_results)}/{len(results)} images successfully")
            print(f"{param_display_name} X - MAE: {np.mean(np.abs(errors_x)):.4f}, RMSE: {np.sqrt(np.mean([e**2 for e in errors_x])):.4f}")
            print(f"{param_display_name} Y - MAE: {np.mean(np.abs(errors_y)):.4f}, RMSE: {np.sqrt(np.mean([e**2 for e in errors_y])):.4f}")
            print(f"Visualization saved to: {output_path}")


def create_common_argument_parser(param_name: str, param_display_name: str, default_output: str):
    """
    Create a common argument parser for effectiveness visualization scripts.
    
    Args:
        param_name: Parameter name (e.g., 'scale', 'offset')
        param_display_name: Parameter display name (e.g., 'Scale', 'Offset')
        default_output: Default output filename
        
    Returns:
        Configured ArgumentParser
    """
    import argparse
    
    parser = argparse.ArgumentParser(description=f"Visualize {param_name} detection model effectiveness")
    
    parser.add_argument("--image", type=str, required=True,
                       help="Input image path")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained scale/offset detection model")
    parser.add_argument("--output", type=str, default=default_output,
                       help="Output visualization path")
    
    # Parameter testing parameters
    parser.add_argument(f"--{param_name}_min", type=float, 
                       help=f"Minimum {param_name} factor")
    parser.add_argument(f"--{param_name}_max", type=float,
                       help=f"Maximum {param_name} factor")
    parser.add_argument("--num_steps", type=int, default=9,
                       help=f"Number of {param_name} steps in each direction")
    
    # System parameters
    parser.add_argument("--input_size", type=int, default=128,
                       help="Model input size")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--temp_dir", type=str,
                       help=f"Temporary directory for {param_name} images")
    parser.add_argument("--keep_temp", action="store_true",
                       help="Keep temporary files after processing")
    
    return parser


def validate_and_run_analysis(args, visualizer, param_name: str, param_display_name: str, default_range: Tuple[float, float]):
    """
    Common validation and analysis execution logic.
    
    Args:
        args: Parsed command line arguments
        visualizer: Visualizer instance
        param_name: Parameter name (e.g., 'scale', 'offset')
        param_display_name: Parameter display name (e.g., 'Scale', 'Offset')
        default_range: Default parameter range
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    # Get parameter range
    param_min = getattr(args, f"{param_name}_min")
    param_max = getattr(args, f"{param_name}_max")
    
    if param_min is None:
        param_min = default_range[0]
    if param_max is None:
        param_max = default_range[1]
    
    param_range = (param_min, param_max)
    
    try:
        # Run analysis
        visualizer.run_analysis(
            image_path=args.image,
            output_path=args.output,
            param_range=param_range,
            num_steps=args.num_steps,
            temp_dir=args.temp_dir,
            cleanup=not args.keep_temp,
            param_name=param_name,
            param_display_name=param_display_name,
            batch_size=args.batch_size
        )
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1