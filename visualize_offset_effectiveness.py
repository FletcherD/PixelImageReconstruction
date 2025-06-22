#!/usr/bin/env python3
"""
Script to visualize the effectiveness of the offset detection model.
Takes an image, applies various offset transformations, runs inference,
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

class OffsetEffectivenessVisualizer(EffectivenessVisualizerBase):
    """Visualizes the effectiveness of offset detection across different input offsets."""
    
    def create_transformed_images(self, 
                                original_image_path: str,
                                param_range: Tuple[float, float] = (-0.5, 0.5),
                                num_steps: int = 9,
                                temp_dir: Optional[str] = None) -> List[Tuple[float, float, str]]:
        """
        Create offset versions of the input image.
        
        Args:
            original_image_path: Path to the original image
            param_range: (min_offset, max_offset) for both axes
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
        offsets = np.linspace(param_range[0], param_range[1], num_steps)
        
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

    
    
    


def main():
    parser = create_common_argument_parser(
        param_name="offset",
        param_display_name="Offset", 
        default_output="offset_effectiveness_visualization.png"
    )
    
    # Set default values for offset parameters
    parser.set_defaults(offset_min=-0.5, offset_max=0.5)
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = OffsetEffectivenessVisualizer(
        model_path=args.model_path,
        input_size=args.input_size
    )
    
    return validate_and_run_analysis(
        args=args,
        visualizer=visualizer,
        param_name="offset",
        param_display_name="Offset",
        default_range=(-0.5, 0.5)
    )


if __name__ == "__main__":
    exit(main())