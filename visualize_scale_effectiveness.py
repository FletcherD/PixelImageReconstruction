#!/usr/bin/env python3
"""
Script to visualize the effectiveness of the scale detection model.
Takes an image, scales it across a range of factors, runs inference,
and creates a visualization of the results.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

from effectiveness_visualization_base import (
    EffectivenessVisualizerBase, 
    create_common_argument_parser,
    validate_and_run_analysis
)
from pixel_spacing_optimizer import rescale_image

class ScaleEffectivenessVisualizer(EffectivenessVisualizerBase):
    """Visualizes the effectiveness of scale detection across different input scales."""
    
    def create_transformed_images(self, 
                                original_image_path: str,
                                param_range: Tuple[float, float] = (0.8, 1.2),
                                num_steps: int = 9,
                                temp_dir: Optional[str] = None) -> List[Tuple[float, float, str]]:
        """
        Create scaled versions of the input image.
        
        Args:
            original_image_path: Path to the original image
            param_range: (min_scale, max_scale) for both axes
            num_steps: Number of scale steps in each direction
            temp_dir: Temporary directory for scaled images
            
        Returns:
            List of (scale_x, scale_y, image_path) tuples
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="scale_test_")
        else:
            os.makedirs(temp_dir, exist_ok=True)
        
        print(f"Creating scaled images in {temp_dir}")
        
        # Load original image
        original_image = Image.open(original_image_path).convert('RGB')
        original_width, original_height = original_image.size
        
        # Generate scale factors
        scales = np.linspace(param_range[0], param_range[1], num_steps)
        
        scaled_images = []
        
        for i, scale_x in enumerate(scales):
            for j, scale_y in enumerate(scales):
                # Calculate new dimensions
                
                # Resize image
                scaled_image = rescale_image(original_image, scale_x, scale_y, 0, 0)
                
                # Save scaled image
                scaled_path = os.path.join(temp_dir, f"scaled_{i:02d}_{j:02d}_sx{scale_x:.2f}_sy{scale_y:.2f}.png")
                scaled_image.save(scaled_path)
                
                scaled_images.append((scale_x, scale_y, scaled_path))
        
        print(f"Created {len(scaled_images)} scaled images")
        return scaled_images

    
    def run_inference_on_images(self, 
                              transformed_images: List[Tuple[float, float, str]],
                              param_name: str) -> List[Dict]:
        """
        Run scale inference on all scaled images.
        
        Args:
            transformed_images: List of (scale_x, scale_y, image_path) tuples
            param_name: Parameter name ('scale')
            
        Returns:
            List of inference results with ground truth scales
        """
        print("Running inference on scaled images...")
        
        results = []
        
        for true_param_x, true_param_y, image_path in transformed_images:
            try:
                # Run inference
                result = self.inference.predict_single(image_path)
                
                # Add ground truth information
                result[f'true_{param_name}_x'] = true_param_x
                result[f'true_{param_name}_y'] = true_param_y
                result['image_path'] = image_path
                
                # Calculate errors
                result[f'error_{param_name}_x'] = result[f'{param_name}_x'] - true_param_x
                result[f'error_{param_name}_y'] = result[f'{param_name}_y'] - true_param_y
                result[f'abs_error_{param_name}_x'] = abs(result[f'error_{param_name}_x'])
                result[f'abs_error_{param_name}_y'] = abs(result[f'error_{param_name}_y'])
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    f'true_{param_name}_x': true_param_x,
                    f'true_{param_name}_y': true_param_y,
                    'image_path': image_path,
                    'error': str(e)
                })
        
        valid_results = [r for r in results if 'error' not in r]
        print(f"Successfully processed {len(valid_results)}/{len(results)} images")
        
        return results
    
    


def main():
    parser = create_common_argument_parser(
        param_name="scale",
        param_display_name="Scale", 
        default_output="scale_effectiveness_visualization.png"
    )
    
    # Set default values for scale parameters
    parser.set_defaults(scale_min=0.8, scale_max=1.2)
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ScaleEffectivenessVisualizer(
        model_path=args.model_path,
        input_size=args.input_size
    )
    
    return validate_and_run_analysis(
        args=args,
        visualizer=visualizer,
        param_name="scale",
        param_display_name="Scale",
        default_range=(0.8, 1.2)
    )


if __name__ == "__main__":
    exit(main())
