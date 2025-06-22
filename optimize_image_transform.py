#!/usr/bin/env python3
"""
Gradient Descent Scale and Offset Optimization

This script uses gradient descent to find the optimal scale and/or offset parameters for an image
such that the model predicts values as close to 0 as possible.
A prediction of 0 indicates the image is at the "intended resolution" and alignment for the model.
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from PIL import Image
import torch

# Add ml-training to path to import the inference module
sys.path.append(str(Path(__file__).parent / "ml-training"))
from infer_scale_offset import ScaleOffsetInference

from pixel_spacing_optimizer import rescale_image

class ImageParameterOptimizer:
    """Gradient descent optimizer for finding optimal image scale and/or offset parameters."""
    
    def __init__(self, 
                 model_path: str,
                 input_size: int = 128,
                 learning_rate: float = 0.01,
                 tolerance: float = 1e-4,
                 max_iterations: int = 100):
        """
        Initialize the parameter optimizer.
        
        Args:
            model_path: Path to the trained scale/offset detection model
            input_size: Input size for the model
            learning_rate: Initial learning rate for gradient descent
            tolerance: Convergence tolerance
            max_iterations: Maximum number of optimization iterations
        """
        self.model_path = model_path
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # Initialize inference engine
        print("Loading scale/offset detection model...")
        self.inference = ScaleOffsetInference(
            model_path=model_path,
            input_size=input_size
        )
        print("Model loaded successfully!")
        
        # Optimization history - will be populated based on optimization mode
        self.history = {}

    def _initialize_history(self, optimize_scale: bool, optimize_offset: bool):
        """Initialize optimization history based on what we're optimizing."""
        self.history = {
            'loss': [],
            'learning_rates': []
        }
        
        if optimize_scale:
            self.history.update({
                'scale_x': [],
                'scale_y': [], 
                'pred_scale_x': [],
                'pred_scale_y': [],
                'pred_scale_x_std': [],
                'pred_scale_y_std': []
            })
            
        if optimize_offset:
            self.history.update({
                'offset_x': [],
                'offset_y': [],
                'pred_offset_x': [],
                'pred_offset_y': [],
                'pred_offset_x_std': [],
                'pred_offset_y_std': []
            })
    
    def objective_function(self, 
                          scale_x: float, scale_y: float, 
                          offset_x: float, offset_y: float,
                          original_image: Image.Image,
                          optimize_scale: bool = True,
                          optimize_offset: bool = False) -> Tuple[float, Dict]:
        """
        Generalized objective function to minimize predicted parameters.
        
        Args:
            scale_x: Current X scaling factor
            scale_y: Current Y scaling factor
            offset_x: Current X offset
            offset_y: Current Y offset
            original_image: Original PIL Image
            optimize_scale: Whether to include scale in loss
            optimize_offset: Whether to include offset in loss
            
        Returns:
            Tuple of (loss_value, prediction_dict)
        """
        # Apply transformation to image
        transformed_image = rescale_image(original_image, scale_x, scale_y, offset_x, offset_y)
        
        # Get predictions using patch-based inference
        result = self.inference.predict_single(transformed_image, use_patches=True)
        
        # Calculate loss based on what we're optimizing
        loss = 0.0
        
        if optimize_scale:
            pred_scale_x = result['scale_x']
            pred_scale_y = result['scale_y']
            loss += pred_scale_x**2 + pred_scale_y**2
            
        if optimize_offset:
            pred_offset_x = result['offset_x'] 
            pred_offset_y = result['offset_y']
            loss += pred_offset_x**2 + pred_offset_y**2
        
        return loss, result
    
    def compute_numerical_gradient(self, 
                                  scale_x: float, scale_y: float,
                                  offset_x: float, offset_y: float,
                                  original_image: Image.Image,
                                  optimize_scale: bool = True,
                                  optimize_offset: bool = False,
                                  epsilon: float = 0.001) -> Tuple[float, float, float, float]:
        """
        Compute numerical gradient of the objective function.
        
        Args:
            scale_x: Current X scaling factor
            scale_y: Current Y scaling factor
            offset_x: Current X offset
            offset_y: Current Y offset
            original_image: Original PIL Image
            optimize_scale: Whether we're optimizing scale
            optimize_offset: Whether we're optimizing offset
            epsilon: Small value for numerical differentiation
            
        Returns:
            Tuple of (grad_scale_x, grad_scale_y, grad_offset_x, grad_offset_y)
        """
        grad_scale_x = grad_scale_y = grad_offset_x = grad_offset_y = 0.0
        
        if optimize_scale:
            # Compute partial derivative w.r.t. scale_x
            loss_plus_x, _ = self.objective_function(scale_x + epsilon, scale_y, offset_x, offset_y, original_image, optimize_scale, optimize_offset)
            loss_minus_x, _ = self.objective_function(scale_x - epsilon, scale_y, offset_x, offset_y, original_image, optimize_scale, optimize_offset)
            grad_scale_x = (loss_plus_x - loss_minus_x) / (2 * epsilon)
            
            # Compute partial derivative w.r.t. scale_y
            loss_plus_y, _ = self.objective_function(scale_x, scale_y + epsilon, offset_x, offset_y, original_image, optimize_scale, optimize_offset)
            loss_minus_y, _ = self.objective_function(scale_x, scale_y - epsilon, offset_x, offset_y, original_image, optimize_scale, optimize_offset)
            grad_scale_y = (loss_plus_y - loss_minus_y) / (2 * epsilon)
        
        if optimize_offset:
            # Compute partial derivative w.r.t. offset_x
            loss_plus_ox, _ = self.objective_function(scale_x, scale_y, offset_x + epsilon, offset_y, original_image, optimize_scale, optimize_offset)
            loss_minus_ox, _ = self.objective_function(scale_x, scale_y, offset_x - epsilon, offset_y, original_image, optimize_scale, optimize_offset)
            grad_offset_x = (loss_plus_ox - loss_minus_ox) / (2 * epsilon)
            
            # Compute partial derivative w.r.t. offset_y
            loss_plus_oy, _ = self.objective_function(scale_x, scale_y, offset_x, offset_y + epsilon, original_image, optimize_scale, optimize_offset)
            loss_minus_oy, _ = self.objective_function(scale_x, scale_y, offset_x, offset_y - epsilon, original_image, optimize_scale, optimize_offset)
            grad_offset_y = (loss_plus_oy - loss_minus_oy) / (2 * epsilon)
        
        return grad_scale_x, grad_scale_y, grad_offset_x, grad_offset_y
    
    def optimize(self, 
                image_path: str,
                initial_scale: Tuple[float, float] = (1.0, 1.0),
                initial_offset: Tuple[float, float] = (0.0, 0.0),
                optimize_scale: bool = True,
                optimize_offset: bool = False,
                adaptive_lr: bool = True,
                verbose: bool = True) -> Dict:
        """
        Optimize the image parameters using gradient descent.
        
        Args:
            image_path: Path to the image to optimize
            initial_scale: Initial (scale_x, scale_y) values
            initial_offset: Initial (offset_x, offset_y) values
            optimize_scale: Whether to optimize scale parameters
            optimize_offset: Whether to optimize offset parameters
            adaptive_lr: Whether to use adaptive learning rate
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results
        """
        # Validate optimization mode
        if not optimize_scale and not optimize_offset:
            raise ValueError("Must optimize at least one of scale or offset")
            
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Initialize parameters
        scale_x, scale_y = initial_scale
        offset_x, offset_y = initial_offset
        current_lr = self.learning_rate
        
        # Initialize history based on optimization mode
        self._initialize_history(optimize_scale, optimize_offset)
        
        if verbose:
            print(f"Starting optimization for {image_path}")
            if optimize_scale:
                print(f"Initial scale: ({scale_x:.4f}, {scale_y:.4f})")
            if optimize_offset:
                print(f"Initial offset: ({offset_x:.4f}, {offset_y:.4f})")
            print(f"Optimizing: {'Scale' if optimize_scale else ''}{',' if optimize_scale and optimize_offset else ''}{'Offset' if optimize_offset else ''}")
            print(f"Target: (0.0000, 0.0000)")
            print("-" * 60)
        
        best_loss = float('inf')
        best_scales = (scale_x, scale_y)
        best_offsets = (offset_x, offset_y)
        patience = 100
        no_improve_count = 0
        
        for iteration in range(self.max_iterations):
            # Evaluate current position
            loss, result = self.objective_function(scale_x, scale_y, offset_x, offset_y, original_image, optimize_scale, optimize_offset)
            
            # Record history
            if optimize_scale:
                self.history['scale_x'].append(scale_x)
                self.history['scale_y'].append(scale_y)
                self.history['pred_scale_x'].append(result['scale_x'])
                self.history['pred_scale_y'].append(result['scale_y'])
                self.history['pred_scale_x_std'].append(result.get('scale_x_std', 0))
                self.history['pred_scale_y_std'].append(result.get('scale_y_std', 0))
            
            if optimize_offset:
                self.history['offset_x'].append(offset_x)
                self.history['offset_y'].append(offset_y)
                self.history['pred_offset_x'].append(result['offset_x'])
                self.history['pred_offset_y'].append(result['offset_y'])
                self.history['pred_offset_x_std'].append(result.get('offset_x_std', 0))
                self.history['pred_offset_y_std'].append(result.get('offset_y_std', 0))
                
            self.history['loss'].append(loss)
            self.history['learning_rates'].append(current_lr)
            
            if verbose:
                output_str = f"Iter {iteration:3d}: "
                if optimize_scale:
                    output_str += f"Scale=({scale_x:.4f}, {scale_y:.4f}) "
                    output_str += f"PredS=({result['scale_x']:+.4f}, {result['scale_y']:+.4f}) "
                if optimize_offset:
                    output_str += f"Offset=({offset_x:.4f}, {offset_y:.4f}) "
                    output_str += f"PredO=({result['offset_x']:+.4f}, {result['offset_y']:+.4f}) "
                output_str += f"Loss={loss:.6f} LR={current_lr:.4f}"
                print(output_str)
            
            # Check for improvement
            if loss < best_loss - self.tolerance:
                best_loss = loss
                best_scales = (scale_x, scale_y)
                best_offsets = (offset_x, offset_y)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= patience:
                if verbose:
                    print(f"Early stopping at iteration {iteration} (no improvement for {patience} steps)")
                break
            
            # Check convergence
            if loss < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration} (loss < {self.tolerance})")
                break
            # Adaptive learning rate
            if adaptive_lr and iteration > 0:
                # Decrease learning rate if loss increased
                if loss > self.history['loss'][-2]:
                    current_lr *= 0.9
                # Increase learning rate if loss decreased significantly
                elif loss < self.history['loss'][-2] * 0.95:
                    current_lr *= 1.05
                
                # Clamp learning rate
                current_lr = np.clip(current_lr, 1e-4, 0.05)
            
            # Update parameters using sign-based heuristic
            if optimize_scale:
                pred_scale_x = result['scale_x']
                pred_scale_y = result['scale_y']
                
                # If predicted scale is negative, increase actual scale, and vice versa
                scale_x -= current_lr * np.sign(pred_scale_x)
                scale_y -= current_lr * np.sign(pred_scale_y)
                
                # Constrain scales to reasonable range
                scale_x = np.clip(scale_x, 0.1, 5.0)
                scale_y = np.clip(scale_y, 0.1, 5.0)
            
            if optimize_offset:
                pred_offset_x = result['offset_x']
                pred_offset_y = result['offset_y']
                
                # If predicted offset is negative, increase actual offset, and vice versa
                offset_x -= current_lr * np.sign(pred_offset_x)
                offset_y -= current_lr * np.sign(pred_offset_y)
                
                # Constrain offsets to reasonable range
                offset_x = np.clip(offset_x, -1.0, 1.0)
                offset_y = np.clip(offset_y, -1.0, 1.0)
        
        # Final evaluation at best point
        final_loss, final_result = self.objective_function(best_scales[0], best_scales[1], best_offsets[0], best_offsets[1], original_image, optimize_scale, optimize_offset)
        
        optimization_result = {
            'success': final_loss < self.tolerance * 10,  # Relaxed success criterion
            'final_scales': best_scales,
            'final_offsets': best_offsets,
            'final_predictions': {
                'scale': (final_result['scale_x'], final_result['scale_y']),
                'offset': (final_result['offset_x'], final_result['offset_y'])
            },
            'final_loss': final_loss,
            'iterations': len(self.history['loss']),
            'history': self.history.copy(),
            'initial_scale': initial_scale,
            'initial_offset': initial_offset,
            'optimize_scale': optimize_scale,
            'optimize_offset': optimize_offset,
            'original_size': original_image.size,
            'optimized_size': (
                int(original_image.size[0] * best_scales[0]),
                int(original_image.size[1] * best_scales[1])
            )
        }
        
        if verbose:
            print("-" * 60)
            print(f"Optimization complete!")
            if optimize_scale:
                print(f"Final scale: ({best_scales[0]:.4f}, {best_scales[1]:.4f})")
                print(f"Final scale predictions: ({final_result['scale_x']:+.4f}, {final_result['scale_y']:+.4f})")
            if optimize_offset:
                print(f"Final offset: ({best_offsets[0]:.4f}, {best_offsets[1]:.4f})")
                print(f"Final offset predictions: ({final_result['offset_x']:+.4f}, {final_result['offset_y']:+.4f})")
            print(f"Final loss: {final_loss:.6f}")
            print(f"Original size: {original_image.size}")
            print(f"Optimized size: {optimization_result['optimized_size']}")
            print(f"Success: {optimization_result['success']}")
        
        return optimization_result
    
    def save_optimized_image(self, 
                           image_path: str, 
                           optimal_scales: Tuple[float, float],
                           optimal_offsets: Tuple[float, float],
                           output_path: str):
        """
        Save the image with optimal scale and offset transformations.
        
        Args:
            image_path: Path to original image
            optimal_scales: Optimal (scale_x, scale_y) values
            optimal_offsets: Optimal (offset_x, offset_y) values
            output_path: Path to save optimized image
        """
        original_image = Image.open(image_path).convert('RGB')
        scale_x, scale_y = optimal_scales
        offset_x, offset_y = optimal_offsets
        
        # Apply transformation using rescale_image function
        optimized_image = rescale_image(original_image, scale_x, scale_y, offset_x, offset_y)
        optimized_image.save(output_path)
        
        print(f"Optimized image saved to {output_path}")
        print(f"Size: {original_image.size} -> {optimized_image.size}")
        print(f"Scale factors: ({scale_x:.4f}, {scale_y:.4f})")
        print(f"Offset values: ({offset_x:.4f}, {offset_y:.4f})")
    
    def visualize_optimization(self, 
                             optimization_result: Dict,
                             output_path: str):
        """
        Create visualization of the optimization process.
        
        Args:
            optimization_result: Result from optimize() method
            output_path: Path to save visualization
        """
        history = optimization_result['history']
        optimize_scale = optimization_result['optimize_scale']
        optimize_offset = optimization_result['optimize_offset']
        
        # Determine subplot layout based on what was optimized
        if optimize_scale and optimize_offset:
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
        
        iterations = range(len(history['loss']))
        plot_idx = 0
        
        # Loss over time
        ax = axes[plot_idx]
        ax.plot(iterations, history['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (Prediction²)')
        ax.set_title('Optimization Loss')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plot_idx += 1
        
        # Scale plots
        if optimize_scale:
            # Scale factors over time
            ax = axes[plot_idx]
            ax.plot(iterations, history['scale_x'], 'r-', label='Scale X', linewidth=2)
            ax.plot(iterations, history['scale_y'], 'g-', label='Scale Y', linewidth=2)
            ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Original Scale')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Scale Factor')
            ax.set_title('Scale Factor Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
            # Scale predictions over time
            ax = axes[plot_idx]
            ax.plot(iterations, history['pred_scale_x'], 'r-', label='Predicted Scale X', linewidth=2)
            ax.plot(iterations, history['pred_scale_y'], 'g-', label='Predicted Scale Y', linewidth=2)
            ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='Target')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Predicted Scale')
            ax.set_title('Scale Model Predictions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Offset plots
        if optimize_offset:
            # Offset factors over time
            ax = axes[plot_idx]
            ax.plot(iterations, history['offset_x'], 'r-', label='Offset X', linewidth=2)
            ax.plot(iterations, history['offset_y'], 'g-', label='Offset Y', linewidth=2)
            ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='Zero Offset')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Offset Value')
            ax.set_title('Offset Value Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
            # Offset predictions over time
            ax = axes[plot_idx]
            ax.plot(iterations, history['pred_offset_x'], 'r-', label='Predicted Offset X', linewidth=2)
            ax.plot(iterations, history['pred_offset_y'], 'g-', label='Predicted Offset Y', linewidth=2)
            ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='Target')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Predicted Offset')
            ax.set_title('Offset Model Predictions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Learning rate over time
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            ax.plot(iterations, history['learning_rates'], 'purple', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Adaptive Learning Rate')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        # Add summary text
        final_scales = optimization_result['final_scales']
        final_offsets = optimization_result['final_offsets']
        final_preds = optimization_result['final_predictions']
        
        summary_text = f"Optimization Summary:\n"
        if optimize_scale:
            summary_text += f"Initial Scale: ({optimization_result['initial_scale'][0]:.3f}, {optimization_result['initial_scale'][1]:.3f})\n"
            summary_text += f"Final Scale: ({final_scales[0]:.3f}, {final_scales[1]:.3f})\n"
            summary_text += f"Scale Predictions: ({final_preds['scale'][0]:+.3f}, {final_preds['scale'][1]:+.3f})\n"
        if optimize_offset:
            summary_text += f"Initial Offset: ({optimization_result['initial_offset'][0]:.3f}, {optimization_result['initial_offset'][1]:.3f})\n"
            summary_text += f"Final Offset: ({final_offsets[0]:.3f}, {final_offsets[1]:.3f})\n"
            summary_text += f"Offset Predictions: ({final_preds['offset'][0]:+.3f}, {final_preds['offset'][1]:+.3f})\n"
        summary_text += f"Final Loss: {optimization_result['final_loss']:.6f}\n"
        summary_text += f"Iterations: {optimization_result['iterations']}\n"
        summary_text += f"Success: {optimization_result['success']}"
        
        fig.suptitle(f"Parameter Optimization Results\n{summary_text}", 
                    fontsize=10, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Optimization visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimize image scale and/or offset using gradient descent")
    
    parser.add_argument("--image", type=str, required=True,
                       help="Input image path")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained scale/offset detection model")
    parser.add_argument("--output_image", type=str,
                       help="Path to save optimized image")
    parser.add_argument("--output_plot", type=str, default="optimization_plot.png",
                       help="Path to save optimization visualization")
    
    # Optimization mode
    parser.add_argument("--optimize", type=str, choices=["scale", "offset", "both"], default="scale",
                       help="What to optimize: scale, offset, or both")
    
    # Optimization parameters
    parser.add_argument("--initial_scale_x", type=float, default=1.0,
                       help="Initial X scale factor")
    parser.add_argument("--initial_scale_y", type=float, default=1.0,
                       help="Initial Y scale factor")
    parser.add_argument("--initial_offset_x", type=float, default=0.0,
                       help="Initial X offset")
    parser.add_argument("--initial_offset_y", type=float, default=0.0,
                       help="Initial Y offset")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                       help="Initial learning rate")
    parser.add_argument("--max_iterations", type=int, default=100,
                       help="Maximum optimization iterations")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                       help="Convergence tolerance")
    
    # System parameters
    parser.add_argument("--input_size", type=int, default=128,
                       help="Model input size")
    parser.add_argument("--no_adaptive_lr", action="store_true",
                       help="Disable adaptive learning rate")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    # Determine optimization mode
    optimize_scale = args.optimize in ["scale", "both"]
    optimize_offset = args.optimize in ["offset", "both"]
    
    try:
        # Create optimizer
        optimizer = ImageParameterOptimizer(
            model_path=args.model_path,
            input_size=args.input_size,
            learning_rate=args.learning_rate,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations
        )
        
        # Run optimization
        result = optimizer.optimize(
            image_path=args.image,
            initial_scale=(args.initial_scale_x, args.initial_scale_y),
            initial_offset=(args.initial_offset_x, args.initial_offset_y),
            optimize_scale=optimize_scale,
            optimize_offset=optimize_offset,
            adaptive_lr=not args.no_adaptive_lr,
            verbose=not args.quiet
        )
        
        # Save optimized image if requested
        if args.output_image:
            optimizer.save_optimized_image(
                image_path=args.image,
                optimal_scales=result['final_scales'],
                optimal_offsets=result['final_offsets'],
                output_path=args.output_image
            )
        
        # Create visualization
        optimizer.visualize_optimization(result, args.output_plot)
        
        # Print final summary
        if not args.quiet:
            print("\n" + "="*60)
            print("OPTIMIZATION SUMMARY")
            print("="*60)
            print(f"Image: {args.image}")
            print(f"Optimization mode: {args.optimize}")
            print(f"Original size: {result['original_size']}")
            print(f"Optimized size: {result['optimized_size']}")
            if optimize_scale:
                print(f"Scale factors: ({result['final_scales'][0]:.4f}, {result['final_scales'][1]:.4f})")
                print(f"Final scale predictions: ({result['final_predictions']['scale'][0]:+.4f}, {result['final_predictions']['scale'][1]:+.4f})")
            if optimize_offset:
                print(f"Offset values: ({result['final_offsets'][0]:.4f}, {result['final_offsets'][1]:.4f})")
                print(f"Final offset predictions: ({result['final_predictions']['offset'][0]:+.4f}, {result['final_predictions']['offset'][1]:+.4f})")
            print(f"Final loss: {result['final_loss']:.6f}")
            print(f"Converged: {result['success']}")
            print(f"Iterations: {result['iterations']}")
        
        return 0
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
