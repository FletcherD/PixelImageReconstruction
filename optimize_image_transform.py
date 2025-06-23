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
                 max_iterations: int = 100,
                 scale_model_path: str = None,
                 offset_model_path: str = None):
        """
        Initialize the parameter optimizer.
        
        Args:
            model_path: Path to the trained scale/offset detection model (fallback if separate models not specified)
            input_size: Input size for the model
            learning_rate: Initial learning rate for gradient descent
            tolerance: Convergence tolerance
            max_iterations: Maximum number of optimization iterations
            scale_model_path: Optional separate model for scale predictions
            offset_model_path: Optional separate model for offset predictions
        """
        self.model_path = model_path
        self.scale_model_path = scale_model_path or model_path
        self.offset_model_path = offset_model_path or model_path
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # Initialize inference engines
        print("Loading scale/offset detection model(s)...")
        
        # Main inference engine (for combined predictions or fallback)
        self.inference = ScaleOffsetInference(
            model_path=model_path,
            input_size=input_size
        )
        
        # Separate scale model if specified
        if scale_model_path and scale_model_path != model_path:
            print(f"Loading separate scale model: {scale_model_path}")
            self.scale_inference = ScaleOffsetInference(
                model_path=scale_model_path,
                input_size=input_size
            )
        else:
            self.scale_inference = self.inference
            
        # Separate offset model if specified  
        if offset_model_path and offset_model_path != model_path:
            print(f"Loading separate offset model: {offset_model_path}")
            self.offset_inference = ScaleOffsetInference(
                model_path=offset_model_path,
                input_size=input_size
            )
        else:
            self.offset_inference = self.inference
            
        print("Model(s) loaded successfully!")
        
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
    
    def variance_objective_function(self, 
                                   scale_x: float, scale_y: float, 
                                   offset_x: float, offset_y: float,
                                   original_image: Image.Image) -> Tuple[float, Dict]:
        """
        Objective function that minimizes the variance of predicted offsets across patches.
        
        Args:
            scale_x: Current X scaling factor
            scale_y: Current Y scaling factor
            offset_x: Current X offset
            offset_y: Current Y offset
            original_image: Original PIL Image
            
        Returns:
            Tuple of (variance_loss, prediction_dict)
        """
        # Apply transformation to image
        transformed_image = rescale_image(original_image, scale_x, scale_y, offset_x, offset_y)
        
        # Get predictions using appropriate models
        offset_result = self.offset_inference.predict_single(transformed_image, use_patches=True)
        scale_result = self.scale_inference.predict_single(transformed_image, use_patches=True)
        
        # Combine results for consistency with existing code
        result = {
            'offset_x': offset_result['offset_x'],
            'offset_y': offset_result['offset_y'],
            'scale_x': scale_result['scale_x'],
            'scale_y': scale_result['scale_y'],
            'offset_x_std': offset_result.get('offset_x_std', 0.0),
            'offset_y_std': offset_result.get('offset_y_std', 0.0),
            'scale_x_std': scale_result.get('scale_x_std', 0.0),
            'scale_y_std': scale_result.get('scale_y_std', 0.0),
        }
        
        # Calculate variance of offset predictions across patches using offset model
        if hasattr(self.offset_inference, 'last_patch_results') and self.offset_inference.last_patch_results:
            offset_x_preds = [patch['offset_x'] for patch in self.offset_inference.last_patch_results]
            offset_y_preds = [patch['offset_y'] for patch in self.offset_inference.last_patch_results]
            
            var_offset_x = np.var(offset_x_preds)
            var_offset_y = np.var(offset_y_preds)
            variance_loss = var_offset_x + var_offset_y
        else:
            # Fallback: use standard deviation from result if available
            var_offset_x = result.get('offset_x_std', 0.0) ** 2
            var_offset_y = result.get('offset_y_std', 0.0) ** 2
            variance_loss = var_offset_x + var_offset_y
        
        return variance_loss, result

    def variance_x_objective(self, scale_x: float, offset_x: float, 
                            scale_y: float, offset_y: float, 
                            original_image: Image.Image) -> float:
        """Objective function for X variance only."""
        transformed_image = rescale_image(original_image, scale_x, scale_y, offset_x, offset_y)
        result = self.offset_inference.predict_single(transformed_image, use_patches=True)
        
        if hasattr(self.offset_inference, 'last_patch_results') and self.offset_inference.last_patch_results:
            offset_x_preds = [patch['offset_x'] for patch in self.offset_inference.last_patch_results]
            return np.var(offset_x_preds)
        else:
            return result.get('offset_x_std', 0.0) ** 2

    def variance_y_objective(self, scale_y: float, offset_y: float,
                            scale_x: float, offset_x: float,
                            original_image: Image.Image) -> float:
        """Objective function for Y variance only."""
        transformed_image = rescale_image(original_image, scale_x, scale_y, offset_x, offset_y)
        result = self.offset_inference.predict_single(transformed_image, use_patches=True)
        
        if hasattr(self.offset_inference, 'last_patch_results') and self.offset_inference.last_patch_results:
            offset_y_preds = [patch['offset_y'] for patch in self.offset_inference.last_patch_results]
            return np.var(offset_y_preds)
        else:
            return result.get('offset_y_std', 0.0) ** 2

    def optimize_variance(self, 
                         image_path: str,
                         initial_scale: Tuple[float, float] = (1.0, 1.0),
                         initial_offset: Tuple[float, float] = (0.0, 0.0),
                         optimizer_type: str = "adam",
                         scale_prediction_weight: float = 0.1,
                         offset_prediction_weight: float = 0.1,
                         scale_lr_multiplier: float = 1.0,
                         offset_lr_multiplier: float = 1.0,
                         verbose: bool = True) -> Dict:
        """
        Optimize both scale and offset to minimize variance of predicted offsets across patches.
        
        Args:
            image_path: Path to the image to optimize
            initial_scale: Initial (scale_x, scale_y) values
            initial_offset: Initial (offset_x, offset_y) values
            optimizer_type: Type of optimizer ("sgd", "momentum", "adam", "rmsprop", "adagrad")
            scale_prediction_weight: Weight for scale prediction guidance term
            offset_prediction_weight: Weight for offset prediction guidance term
            scale_lr_multiplier: Learning rate multiplier for scale parameters
            offset_lr_multiplier: Learning rate multiplier for offset parameters
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results
        """
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Initialize parameters
        scale_x, scale_y = initial_scale
        offset_x, offset_y = initial_offset
        current_lr = self.learning_rate
        
        # Initialize optimizer-specific state variables
        if optimizer_type == "momentum":
            # Momentum terms
            v_scale_x = v_scale_y = v_offset_x = v_offset_y = 0.0
            momentum = 0.9
        elif optimizer_type == "adam":
            # Adam first and second moment estimates
            m_scale_x = m_scale_y = m_offset_x = m_offset_y = 0.0
            v_scale_x = v_scale_y = v_offset_x = v_offset_y = 0.0
            beta1, beta2 = 0.9, 0.999
            epsilon_adam = 1e-8
        elif optimizer_type == "rmsprop":
            # RMSprop moving average of squared gradients
            v_scale_x = v_scale_y = v_offset_x = v_offset_y = 0.0
            decay_rate = 0.9
            epsilon_rms = 1e-8
        elif optimizer_type == "adagrad":
            # Adagrad accumulated squared gradients
            G_scale_x = G_scale_y = G_offset_x = G_offset_y = 0.0
            epsilon_ada = 1e-8
        
        # Initialize history for variance optimization
        self.history = {
            'loss': [],
            'learning_rates': [],
            'scale_x': [],
            'scale_y': [],
            'offset_x': [],
            'offset_y': [],
            'pred_offset_x': [],
            'pred_offset_y': [],
            'pred_offset_x_std': [],
            'pred_offset_y_std': [],
            'offset_variance_x': [],
            'offset_variance_y': [],
            'pred_scale_x': [],
            'pred_scale_y': [],
            'scale_guidance_x': [],
            'scale_guidance_y': [],
            'offset_guidance_x': [],
            'offset_guidance_y': []
        }
        
        if verbose:
            print(f"Starting variance-based optimization for {image_path}")
            print(f"Optimizer: {optimizer_type.upper()}")
            print(f"Scale prediction weight: {scale_prediction_weight}")
            print(f"Offset prediction weight: {offset_prediction_weight}")
            print(f"Scale learning rate multiplier: {scale_lr_multiplier}")
            print(f"Offset learning rate multiplier: {offset_lr_multiplier}")
            print(f"Scale model: {self.scale_model_path}")
            print(f"Offset model: {self.offset_model_path}")
            print(f"Initial scale: ({scale_x:.4f}, {scale_y:.4f})")
            print(f"Initial offset: ({offset_x:.4f}, {offset_y:.4f})")
            print(f"Target: minimize variance of offset predictions across patches")
            print("-" * 60)
        
        best_loss = float('inf')
        best_scales = (scale_x, scale_y)
        best_offsets = (offset_x, offset_y)
        patience = 100
        no_improve_count = 0
        
        for iteration in range(self.max_iterations):
            # Evaluate current position using variance objective
            loss, result = self.variance_objective_function(scale_x, scale_y, offset_x, offset_y, original_image)
            
            # Calculate individual variances for tracking
            if hasattr(self.inference, 'last_patch_results') and self.inference.last_patch_results:
                offset_x_preds = [patch['offset_x'] for patch in self.inference.last_patch_results]
                offset_y_preds = [patch['offset_y'] for patch in self.inference.last_patch_results]
                var_offset_x = np.var(offset_x_preds)
                var_offset_y = np.var(offset_y_preds)
            else:
                var_offset_x = result.get('offset_x_std', 0.0) ** 2
                var_offset_y = result.get('offset_y_std', 0.0) ** 2
            
            # Record history
            self.history['scale_x'].append(scale_x)
            self.history['scale_y'].append(scale_y)
            self.history['offset_x'].append(offset_x)
            self.history['offset_y'].append(offset_y)
            self.history['pred_offset_x'].append(result['offset_x'])
            self.history['pred_offset_y'].append(result['offset_y'])
            self.history['pred_offset_x_std'].append(result.get('offset_x_std', 0))
            self.history['pred_offset_y_std'].append(result.get('offset_y_std', 0))
            self.history['offset_variance_x'].append(var_offset_x)
            self.history['offset_variance_y'].append(var_offset_y)
            self.history['pred_scale_x'].append(result['scale_x'])
            self.history['pred_scale_y'].append(result['scale_y'])
            self.history['scale_guidance_x'].append(scale_prediction_weight * result['scale_x'])
            self.history['scale_guidance_y'].append(scale_prediction_weight * result['scale_y'])
            self.history['offset_guidance_x'].append(offset_prediction_weight * result['offset_x'])
            self.history['offset_guidance_y'].append(offset_prediction_weight * result['offset_y'])
            self.history['loss'].append(loss)
            self.history['learning_rates'].append(current_lr)
            
            if verbose:
                print(f"Iter {iteration:3d}: Scale=({scale_x:.4f}, {scale_y:.4f}) "
                      f"Offset=({offset_x:.4f}, {offset_y:.4f}) "
                      f"PredScale=({result['scale_x']:+.3f}, {result['scale_y']:+.3f}) "
                      f"OffsetVar=({var_offset_x:.6f}, {var_offset_y:.6f}) "
                      f"Loss={loss:.6f} LR={current_lr:.4f}")
            
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
                    print(f"Converged at iteration {iteration} (variance < {self.tolerance})")
                break
            
            # Adaptive learning rate
            if iteration > 0:
                # Decrease learning rate if loss increased
                if loss > self.history['loss'][-2]:
                    current_lr *= 0.9
                # Increase learning rate if loss decreased significantly
                elif loss < self.history['loss'][-2] * 0.95:
                    current_lr *= 1.05
                
                # Clamp learning rate
                current_lr = np.clip(current_lr, 1e-4, 0.05)
            
            # Compute numerical gradients for independent X and Y problems
            epsilon = 0.001
            
            # X component: gradient w.r.t. scale_x and offset_x for X variance only
            var_x_plus_sx = self.variance_x_objective(scale_x + epsilon, offset_x, scale_y, offset_y, original_image)
            var_x_minus_sx = self.variance_x_objective(scale_x - epsilon, offset_x, scale_y, offset_y, original_image)
            var_x_grad_scale_x = (var_x_plus_sx - var_x_minus_sx) / (2 * epsilon)
            
            var_x_plus_ox = self.variance_x_objective(scale_x, offset_x + epsilon, scale_y, offset_y, original_image)
            var_x_minus_ox = self.variance_x_objective(scale_x, offset_x - epsilon, scale_y, offset_y, original_image)
            var_x_grad_offset_x = (var_x_plus_ox - var_x_minus_ox) / (2 * epsilon)
            
            # Y component: gradient w.r.t. scale_y and offset_y for Y variance only
            var_y_plus_sy = self.variance_y_objective(scale_y + epsilon, offset_y, scale_x, offset_x, original_image)
            var_y_minus_sy = self.variance_y_objective(scale_y - epsilon, offset_y, scale_x, offset_x, original_image)
            var_y_grad_scale_y = (var_y_plus_sy - var_y_minus_sy) / (2 * epsilon)
            
            var_y_plus_oy = self.variance_y_objective(scale_y, offset_y + epsilon, scale_x, offset_x, original_image)
            var_y_minus_oy = self.variance_y_objective(scale_y, offset_y - epsilon, scale_x, offset_x, original_image)
            var_y_grad_offset_y = (var_y_plus_oy - var_y_minus_oy) / (2 * epsilon)
            
            # Add prediction guidance terms
            # If predicted scale is negative, we want to increase actual scale (negative gradient)
            # If predicted scale is positive, we want to decrease actual scale (positive gradient)
            pred_scale_x = result['scale_x']
            pred_scale_y = result['scale_y']
            scale_guidance_x = scale_prediction_weight * pred_scale_x
            scale_guidance_y = scale_prediction_weight * pred_scale_y
            
            # If predicted offset is negative, we want to increase actual offset (negative gradient)
            # If predicted offset is positive, we want to decrease actual offset (positive gradient)
            pred_offset_x = result['offset_x']
            pred_offset_y = result['offset_y']
            offset_guidance_x = offset_prediction_weight * pred_offset_x
            offset_guidance_y = offset_prediction_weight * pred_offset_y
            
            # Combine variance gradients with prediction guidance
            combined_grad_scale_x = var_x_grad_scale_x + scale_guidance_x
            combined_grad_scale_y = var_y_grad_scale_y + scale_guidance_y
            combined_grad_offset_x = var_x_grad_offset_x + offset_guidance_x
            combined_grad_offset_y = var_y_grad_offset_y + offset_guidance_y
            
            # Update parameters using selected optimizer (treating X and Y independently)
            # Apply learning rate multipliers
            scale_lr = current_lr * scale_lr_multiplier
            offset_lr = current_lr * offset_lr_multiplier
            
            if optimizer_type == "sgd":
                # Standard SGD
                scale_x -= scale_lr * combined_grad_scale_x
                scale_y -= scale_lr * combined_grad_scale_y
                offset_x -= offset_lr * combined_grad_offset_x
                offset_y -= offset_lr * combined_grad_offset_y
                
            elif optimizer_type == "momentum":
                # SGD with momentum
                v_scale_x = momentum * v_scale_x + scale_lr * combined_grad_scale_x
                v_scale_y = momentum * v_scale_y + scale_lr * combined_grad_scale_y
                v_offset_x = momentum * v_offset_x + offset_lr * combined_grad_offset_x
                v_offset_y = momentum * v_offset_y + offset_lr * combined_grad_offset_y
                
                scale_x -= v_scale_x
                scale_y -= v_scale_y
                offset_x -= v_offset_x
                offset_y -= v_offset_y
                
            elif optimizer_type == "adam":
                # Adam optimizer
                m_scale_x = beta1 * m_scale_x + (1 - beta1) * combined_grad_scale_x
                m_scale_y = beta1 * m_scale_y + (1 - beta1) * combined_grad_scale_y
                m_offset_x = beta1 * m_offset_x + (1 - beta1) * combined_grad_offset_x
                m_offset_y = beta1 * m_offset_y + (1 - beta1) * combined_grad_offset_y
                
                v_scale_x = beta2 * v_scale_x + (1 - beta2) * combined_grad_scale_x**2
                v_scale_y = beta2 * v_scale_y + (1 - beta2) * combined_grad_scale_y**2
                v_offset_x = beta2 * v_offset_x + (1 - beta2) * combined_grad_offset_x**2
                v_offset_y = beta2 * v_offset_y + (1 - beta2) * combined_grad_offset_y**2
                
                # Bias correction
                m_scale_x_hat = m_scale_x / (1 - beta1**(iteration + 1))
                m_scale_y_hat = m_scale_y / (1 - beta1**(iteration + 1))
                m_offset_x_hat = m_offset_x / (1 - beta1**(iteration + 1))
                m_offset_y_hat = m_offset_y / (1 - beta1**(iteration + 1))
                
                v_scale_x_hat = v_scale_x / (1 - beta2**(iteration + 1))
                v_scale_y_hat = v_scale_y / (1 - beta2**(iteration + 1))
                v_offset_x_hat = v_offset_x / (1 - beta2**(iteration + 1))
                v_offset_y_hat = v_offset_y / (1 - beta2**(iteration + 1))
                
                scale_x -= scale_lr * m_scale_x_hat / (np.sqrt(v_scale_x_hat) + epsilon_adam)
                scale_y -= scale_lr * m_scale_y_hat / (np.sqrt(v_scale_y_hat) + epsilon_adam)
                offset_x -= offset_lr * m_offset_x_hat / (np.sqrt(v_offset_x_hat) + epsilon_adam)
                offset_y -= offset_lr * m_offset_y_hat / (np.sqrt(v_offset_y_hat) + epsilon_adam)
                
            elif optimizer_type == "rmsprop":
                # RMSprop
                v_scale_x = decay_rate * v_scale_x + (1 - decay_rate) * combined_grad_scale_x**2
                v_scale_y = decay_rate * v_scale_y + (1 - decay_rate) * combined_grad_scale_y**2
                v_offset_x = decay_rate * v_offset_x + (1 - decay_rate) * combined_grad_offset_x**2
                v_offset_y = decay_rate * v_offset_y + (1 - decay_rate) * combined_grad_offset_y**2
                
                scale_x -= scale_lr * combined_grad_scale_x / (np.sqrt(v_scale_x) + epsilon_rms)
                scale_y -= scale_lr * combined_grad_scale_y / (np.sqrt(v_scale_y) + epsilon_rms)
                offset_x -= offset_lr * combined_grad_offset_x / (np.sqrt(v_offset_x) + epsilon_rms)
                offset_y -= offset_lr * combined_grad_offset_y / (np.sqrt(v_offset_y) + epsilon_rms)
                
            elif optimizer_type == "adagrad":
                # Adagrad
                G_scale_x += combined_grad_scale_x**2
                G_scale_y += combined_grad_scale_y**2
                G_offset_x += combined_grad_offset_x**2
                G_offset_y += combined_grad_offset_y**2
                
                scale_x -= scale_lr * combined_grad_scale_x / (np.sqrt(G_scale_x) + epsilon_ada)
                scale_y -= scale_lr * combined_grad_scale_y / (np.sqrt(G_scale_y) + epsilon_ada)
                offset_x -= offset_lr * combined_grad_offset_x / (np.sqrt(G_offset_x) + epsilon_ada)
                offset_y -= offset_lr * combined_grad_offset_y / (np.sqrt(G_offset_y) + epsilon_ada)
            
            # Constrain parameters to reasonable ranges
            scale_x = np.clip(scale_x, 0.5, 5.0)
            scale_y = np.clip(scale_y, 0.5, 5.0)
            offset_x = np.clip(offset_x, -1.0, 1.0)
            offset_y = np.clip(offset_y, -1.0, 1.0)
        
        # Final evaluation at best point
        final_loss, final_result = self.variance_objective_function(best_scales[0], best_scales[1], best_offsets[0], best_offsets[1], original_image)
        
        optimization_result = {
            'success': final_loss < self.tolerance * 10,
            'final_scales': best_scales,
            'final_offsets': best_offsets,
            'final_predictions': {
                'offset': (final_result['offset_x'], final_result['offset_y'])
            },
            'final_loss': final_loss,
            'iterations': len(self.history['loss']),
            'history': self.history.copy(),
            'initial_scale': initial_scale,
            'initial_offset': initial_offset,
            'optimize_scale': True,
            'optimize_offset': True,
            'optimization_type': 'variance',
            'optimizer_type': optimizer_type,
            'scale_prediction_weight': scale_prediction_weight,
            'offset_prediction_weight': offset_prediction_weight,
            'scale_lr_multiplier': scale_lr_multiplier,
            'offset_lr_multiplier': offset_lr_multiplier,
            'original_size': original_image.size,
            'optimized_size': (
                int(original_image.size[0] * best_scales[0]),
                int(original_image.size[1] * best_scales[1])
            )
        }
        
        if verbose:
            print("-" * 60)
            print(f"Variance optimization complete!")
            print(f"Final scale: ({best_scales[0]:.4f}, {best_scales[1]:.4f})")
            print(f"Final offset: ({best_offsets[0]:.4f}, {best_offsets[1]:.4f})")
            print(f"Final offset predictions: ({final_result['offset_x']:+.4f}, {final_result['offset_y']:+.4f})")
            print(f"Final variance loss: {final_loss:.6f}")
            print(f"Original size: {original_image.size}")
            print(f"Optimized size: {optimization_result['optimized_size']}")
            print(f"Success: {optimization_result['success']}")
        
        return optimization_result

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
        is_variance_opt = optimization_result.get('optimization_type') == 'variance'
        
        # Determine subplot layout based on what was optimized
        if is_variance_opt:
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            axes = axes.flatten()
        elif optimize_scale and optimize_offset:
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
        
        # Variance-specific plots
        if is_variance_opt and 'offset_variance_x' in history:
            # Offset variance over time
            ax = axes[plot_idx]
            ax.plot(iterations, history['offset_variance_x'], 'r-', label='Offset X Variance', linewidth=2)
            ax.plot(iterations, history['offset_variance_y'], 'g-', label='Offset Y Variance', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Offset Variance')
            ax.set_title('Offset Prediction Variance Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            plot_idx += 1
            
        # Scale prediction guidance plots
        if is_variance_opt and 'pred_scale_x' in history:
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
            
            # Scale guidance terms over time
            if 'scale_guidance_x' in history:
                ax = axes[plot_idx]
                ax.plot(iterations, history['scale_guidance_x'], 'r-', label='Scale X Guidance', linewidth=2)
                ax.plot(iterations, history['scale_guidance_y'], 'g-', label='Scale Y Guidance', linewidth=2)
                ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Scale Guidance Term')
                ax.set_title('Scale Prediction Guidance')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
                
            # Offset guidance terms over time
            if 'offset_guidance_x' in history:
                ax = axes[plot_idx]
                ax.plot(iterations, history['offset_guidance_x'], 'r-', label='Offset X Guidance', linewidth=2)
                ax.plot(iterations, history['offset_guidance_y'], 'g-', label='Offset Y Guidance', linewidth=2)
                ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Offset Guidance Term')
                ax.set_title('Offset Prediction Guidance')
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
        summary_text += f"Type: {'Variance Minimization' if is_variance_opt else 'Prediction Minimization'}\n"
        if optimize_scale:
            summary_text += f"Initial Scale: ({optimization_result['initial_scale'][0]:.3f}, {optimization_result['initial_scale'][1]:.3f})\n"
            summary_text += f"Final Scale: ({final_scales[0]:.3f}, {final_scales[1]:.3f})\n"
            if 'scale' in final_preds:
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
    parser.add_argument("--scale_model_path", type=str,
                       help="Optional separate model for scale predictions")
    parser.add_argument("--offset_model_path", type=str,
                       help="Optional separate model for offset predictions")
    parser.add_argument("--output_image", type=str,
                       help="Path to save optimized image")
    parser.add_argument("--output_plot", type=str, default="optimization_plot.png",
                       help="Path to save optimization visualization")
    
    # Optimization mode
    parser.add_argument("--optimize", type=str, choices=["scale", "offset", "both", "variance"], default="scale",
                       help="What to optimize: scale, offset, both, or variance (minimizes variance of offset predictions)")
    
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
    parser.add_argument("--optimizer", type=str, choices=["sgd", "momentum", "adam", "rmsprop", "adagrad"], default="adam",
                       help="Optimizer type for variance optimization")
    parser.add_argument("--scale_pred_weight", type=float, default=0.1,
                       help="Weight for scale prediction guidance term (variance optimization only)")
    parser.add_argument("--offset_pred_weight", type=float, default=0.1,
                       help="Weight for offset prediction guidance term (variance optimization only)")
    parser.add_argument("--scale_lr_mult", type=float, default=1.0,
                       help="Learning rate multiplier for scale parameters (variance optimization only)")
    parser.add_argument("--offset_lr_mult", type=float, default=1.0,
                       help="Learning rate multiplier for offset parameters (variance optimization only)")
    
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
    
    if args.scale_model_path and not os.path.exists(args.scale_model_path):
        print(f"Error: Scale model file not found: {args.scale_model_path}")
        return 1
        
    if args.offset_model_path and not os.path.exists(args.offset_model_path):
        print(f"Error: Offset model file not found: {args.offset_model_path}")
        return 1
    
    # Determine optimization mode
    optimize_scale = args.optimize in ["scale", "both"]
    optimize_offset = args.optimize in ["offset", "both"]
    is_variance_optimization = args.optimize == "variance"
    
    try:
        # Create optimizer
        optimizer = ImageParameterOptimizer(
            model_path=args.model_path,
            input_size=args.input_size,
            learning_rate=args.learning_rate,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            scale_model_path=args.scale_model_path,
            offset_model_path=args.offset_model_path
        )
        
        # Run optimization
        if is_variance_optimization:
            result = optimizer.optimize_variance(
                image_path=args.image,
                initial_scale=(args.initial_scale_x, args.initial_scale_y),
                initial_offset=(args.initial_offset_x, args.initial_offset_y),
                optimizer_type=args.optimizer,
                scale_prediction_weight=args.scale_pred_weight,
                offset_prediction_weight=args.offset_pred_weight,
                scale_lr_multiplier=args.scale_lr_mult,
                offset_lr_multiplier=args.offset_lr_mult,
                verbose=not args.quiet
            )
        else:
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
            print(f"Scale factors: ({result['final_scales'][0]:.4f}, {result['final_scales'][1]:.4f})")
            print(f"Offset values: ({result['final_offsets'][0]:.4f}, {result['final_offsets'][1]:.4f})")
            if is_variance_optimization:
                print(f"Final offset predictions: ({result['final_predictions']['offset'][0]:+.4f}, {result['final_predictions']['offset'][1]:+.4f})")
                print(f"Final variance loss: {result['final_loss']:.6f}")  
            else:
                if optimize_scale and 'scale' in result['final_predictions']:
                    print(f"Final scale predictions: ({result['final_predictions']['scale'][0]:+.4f}, {result['final_predictions']['scale'][1]:+.4f})")
                if optimize_offset:
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
