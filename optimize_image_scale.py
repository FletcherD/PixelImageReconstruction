#!/usr/bin/env python3
"""
Gradient Descent Scale Optimization

This script uses gradient descent to find the optimal scaling factors for an image
such that the scale detection model predicts values as close to 0 as possible.
A prediction of 0 indicates the image is at the "intended resolution" for the model.
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

class ScaleOptimizer:
    """Gradient descent optimizer for finding optimal image scale."""
    
    def __init__(self, 
                 model_path: str,
                 input_size: int = 128,
                 learning_rate: float = 0.01,
                 tolerance: float = 1e-4,
                 max_iterations: int = 100):
        """
        Initialize the scale optimizer.
        
        Args:
            model_path: Path to the trained scale detection model
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
        print("Loading scale detection model...")
        self.inference = ScaleOffsetInference(
            model_path=model_path,
            input_size=input_size
        )
        print("Model loaded successfully!")
        
        # Optimization history
        self.history = {
            'scale_x': [],
            'scale_y': [],
            'pred_x': [],
            'pred_y': [],
            'pred_x_std': [],
            'pred_y_std': [],
            'loss': [],
            'learning_rates': []
        }

    
    def objective_function(self, scale_x: float, scale_y: float, original_image: Image.Image) -> Tuple[float, Dict]:
        """
        Objective function to minimize: sum of squared predicted scales.
        
        Args:
            scale_x: Current X scaling factor
            scale_y: Current Y scaling factor
            original_image: Original PIL Image
            
        Returns:
            Tuple of (loss_value, prediction_dict)
        """
        # Scale the image
        original_width, original_height = original_image.size
        new_width = int(original_width * scale_x)
        new_height = int(original_height * scale_y)
        
        # Ensure minimum size
        new_width = max(new_width, self.input_size)
        new_height = max(new_height, self.input_size)
        
        scaled_image = rescale_image(original_image, scale_x, scale_y, 0, 0)
        
        # Get predictions using patch-based inference
        result = self.inference.predict_single(scaled_image, use_patches=True)
        
        # Calculate loss: we want both predicted scales to be close to 0
        pred_scale_x = result['scale_x']
        pred_scale_y = result['scale_y']
        
        loss = pred_scale_x**2 + pred_scale_y**2
        
        return loss, result
    
    def compute_numerical_gradient(self, 
                                  scale_x: float, 
                                  scale_y: float, 
                                  original_image: Image.Image,
                                  epsilon: float = 0.001) -> Tuple[float, float]:
        """
        Compute numerical gradient of the objective function.
        
        Args:
            scale_x: Current X scaling factor
            scale_y: Current Y scaling factor
            original_image: Original PIL Image
            epsilon: Small value for numerical differentiation
            
        Returns:
            Tuple of (grad_x, grad_y)
        """
        # Compute partial derivative w.r.t. scale_x
        loss_plus_x, _ = self.objective_function(scale_x + epsilon, scale_y, original_image)
        loss_minus_x, _ = self.objective_function(scale_x - epsilon, scale_y, original_image)
        grad_x = (loss_plus_x - loss_minus_x) / (2 * epsilon)
        
        # Compute partial derivative w.r.t. scale_y
        loss_plus_y, _ = self.objective_function(scale_x, scale_y + epsilon, original_image)
        loss_minus_y, _ = self.objective_function(scale_x, scale_y - epsilon, original_image)
        grad_y = (loss_plus_y - loss_minus_y) / (2 * epsilon)
        
        return grad_x, grad_y
    
    def optimize(self, 
                image_path: str,
                initial_scale: Tuple[float, float] = (1.0, 1.0),
                adaptive_lr: bool = True,
                verbose: bool = True) -> Dict:
        """
        Optimize the image scale using gradient descent.
        
        Args:
            image_path: Path to the image to optimize
            initial_scale: Initial (scale_x, scale_y) values
            adaptive_lr: Whether to use adaptive learning rate
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results
        """
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Initialize parameters
        scale_x, scale_y = initial_scale
        current_lr = self.learning_rate
        
        # Reset history
        for key in self.history:
            self.history[key] = []
        
        if verbose:
            print(f"Starting optimization for {image_path}")
            print(f"Initial scale: ({scale_x:.4f}, {scale_y:.4f})")
            print(f"Target: (0.0000, 0.0000)")
            print("-" * 60)
        
        best_loss = float('inf')
        best_scales = (scale_x, scale_y)
        patience = 100
        no_improve_count = 0
        
        for iteration in range(self.max_iterations):
            # Evaluate current position
            loss, result = self.objective_function(scale_x, scale_y, original_image)
            
            # Record history
            self.history['scale_x'].append(scale_x)
            self.history['scale_y'].append(scale_y)
            self.history['pred_x'].append(result['scale_x'])
            self.history['pred_y'].append(result['scale_y'])
            self.history['pred_x_std'].append(result.get('scale_x_std', 0))
            self.history['pred_y_std'].append(result.get('scale_y_std', 0))
            self.history['loss'].append(loss)
            self.history['learning_rates'].append(current_lr)
            
            if verbose:
                print(f"Iter {iteration:3d}: Scale=({scale_x:.4f}, {scale_y:.4f}) "
                      f"Pred=({result['scale_x']:+.4f}, {result['scale_y']:+.4f}) "
                      f"Loss={loss:.6f} LR={current_lr:.4f}")
            
            # Check for improvement
            if loss < best_loss - self.tolerance:
                best_loss = loss
                best_scales = (scale_x, scale_y)
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
            
            # Compute gradients
            grad_x, grad_y = self.compute_numerical_gradient(scale_x, scale_y, original_image)
            
            # Adaptive learning rate
            if adaptive_lr and iteration > 0:
                # Decrease learning rate if loss increased
                if loss > self.history['loss'][-2]:
                    current_lr *= 0.8
                # Increase learning rate if loss decreased significantly
                elif loss < self.history['loss'][-2] * 0.9:
                    current_lr *= 1.1
                
                # Clamp learning rate
                current_lr = np.clip(current_lr, 1e-6, 0.1)
            
            # Update parameters
            scale_x -= current_lr * grad_x
            scale_y -= current_lr * grad_y
            
            # Constrain scales to reasonable range
            scale_x = np.clip(scale_x, 0.1, 5.0)
            scale_y = np.clip(scale_y, 0.1, 5.0)
        
        # Final evaluation at best point
        final_loss, final_result = self.objective_function(best_scales[0], best_scales[1], original_image)
        
        optimization_result = {
            'success': final_loss < self.tolerance * 10,  # Relaxed success criterion
            'final_scales': best_scales,
            'final_predictions': (final_result['scale_x'], final_result['scale_y']),
            'final_loss': final_loss,
            'iterations': len(self.history['loss']),
            'history': self.history.copy(),
            'initial_scale': initial_scale,
            'original_size': original_image.size,
            'optimized_size': (
                int(original_image.size[0] * best_scales[0]),
                int(original_image.size[1] * best_scales[1])
            )
        }
        
        if verbose:
            print("-" * 60)
            print(f"Optimization complete!")
            print(f"Final scale: ({best_scales[0]:.4f}, {best_scales[1]:.4f})")
            print(f"Final predictions: ({final_result['scale_x']:+.4f}, {final_result['scale_y']:+.4f})")
            print(f"Final loss: {final_loss:.6f}")
            print(f"Original size: {original_image.size}")
            print(f"Optimized size: {optimization_result['optimized_size']}")
            print(f"Success: {optimization_result['success']}")
        
        return optimization_result
    
    def save_optimized_image(self, 
                           image_path: str, 
                           optimal_scales: Tuple[float, float],
                           output_path: str):
        """
        Save the image at optimal scale.
        
        Args:
            image_path: Path to original image
            optimal_scales: Optimal (scale_x, scale_y) values
            output_path: Path to save optimized image
        """
        original_image = Image.open(image_path).convert('RGB')
        scale_x, scale_y = optimal_scales
        
        original_width, original_height = original_image.size
        new_width = int(original_width * scale_x)
        new_height = int(original_height * scale_y)
        
        # Ensure minimum size
        new_width = max(new_width, self.input_size)
        new_height = max(new_height, self.input_size)
        
        optimized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
        optimized_image.save(output_path)
        
        print(f"Optimized image saved to {output_path}")
        print(f"Size: {original_image.size} -> {optimized_image.size}")
        print(f"Scale factors: ({scale_x:.4f}, {scale_y:.4f})")
    
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
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        iterations = range(len(history['loss']))
        
        # Loss over time
        ax = axes[0, 0]
        ax.plot(iterations, history['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (Prediction²)')
        ax.set_title('Optimization Loss')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Scale factors over time
        ax = axes[0, 1]
        ax.plot(iterations, history['scale_x'], 'r-', label='Scale X', linewidth=2)
        ax.plot(iterations, history['scale_y'], 'g-', label='Scale Y', linewidth=2)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Original Scale')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Scale Factor')
        ax.set_title('Scale Factor Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Predictions over time
        ax = axes[0, 2]
        ax.plot(iterations, history['pred_x'], 'r-', label='Predicted X', linewidth=2)
        ax.plot(iterations, history['pred_y'], 'g-', label='Predicted Y', linewidth=2)
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5, label='Target')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Predicted Scale')
        ax.set_title('Model Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate over time
        ax = axes[1, 0]
        ax.plot(iterations, history['learning_rates'], 'purple', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Adaptive Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Prediction uncertainty over time
        ax = axes[1, 1]
        ax.plot(iterations, history['pred_x_std'], 'r-', alpha=0.7, label='Std X')
        ax.plot(iterations, history['pred_y_std'], 'g-', alpha=0.7, label='Std Y')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Prediction Std Dev')
        ax.set_title('Prediction Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2D trajectory in scale space
        ax = axes[1, 2]
        ax.plot(history['scale_x'], history['scale_y'], 'b-', alpha=0.7, linewidth=2)
        ax.scatter(history['scale_x'][0], history['scale_y'][0], 
                  color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(history['scale_x'][-1], history['scale_y'][-1], 
                  color='red', s=100, marker='*', label='End', zorder=5)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Scale X')
        ax.set_ylabel('Scale Y')
        ax.set_title('Optimization Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add summary text
        final_scales = optimization_result['final_scales']
        final_preds = optimization_result['final_predictions']
        
        summary_text = f"""Optimization Summary:
Initial Scale: ({optimization_result['initial_scale'][0]:.3f}, {optimization_result['initial_scale'][1]:.3f})
Final Scale: ({final_scales[0]:.3f}, {final_scales[1]:.3f})
Final Predictions: ({final_preds[0]:+.3f}, {final_preds[1]:+.3f})
Final Loss: {optimization_result['final_loss']:.6f}
Iterations: {optimization_result['iterations']}
Success: {optimization_result['success']}"""
        
        fig.suptitle(f"Scale Optimization Results\n{summary_text}", 
                    fontsize=12, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Optimization visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimize image scale using gradient descent")
    
    parser.add_argument("--image", type=str, required=True,
                       help="Input image path")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained scale detection model")
    parser.add_argument("--output_image", type=str,
                       help="Path to save optimized image")
    parser.add_argument("--output_plot", type=str, default="optimization_plot.png",
                       help="Path to save optimization visualization")
    
    # Optimization parameters
    parser.add_argument("--initial_scale_x", type=float, default=1.0,
                       help="Initial X scale factor")
    parser.add_argument("--initial_scale_y", type=float, default=1.0,
                       help="Initial Y scale factor")
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
    
    try:
        # Create optimizer
        optimizer = ScaleOptimizer(
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
            adaptive_lr=not args.no_adaptive_lr,
            verbose=not args.quiet
        )
        
        # Save optimized image if requested
        if args.output_image:
            optimizer.save_optimized_image(
                image_path=args.image,
                optimal_scales=result['final_scales'],
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
            print(f"Original size: {result['original_size']}")
            print(f"Optimized size: {result['optimized_size']}")
            print(f"Scale factors: ({result['final_scales'][0]:.4f}, {result['final_scales'][1]:.4f})")
            print(f"Final predictions: ({result['final_predictions'][0]:+.4f}, {result['final_predictions'][1]:+.4f})")
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
