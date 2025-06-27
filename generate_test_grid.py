#!/usr/bin/env python3
"""
Grid generation utility for testing pixel reconstruction pipeline.
Generates grid images with sub-pixel accuracy for precise testing.
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw
import sys
import os


def create_subpixel_grid(width, height, grid_size, line_width=1.0, offset_x=0.0, offset_y=0.0, 
                        background_color=(255, 255, 255), line_color=(0, 0, 0)):
    """
    Create a grid image with sub-pixel accuracy.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels  
        grid_size: Grid spacing in pixels (can be fractional)
        line_width: Width of grid lines in pixels (can be fractional)
        offset_x: Horizontal offset of grid in pixels (can be fractional)
        offset_y: Vertical offset of grid in pixels (can be fractional)
        background_color: RGB tuple for background color
        line_color: RGB tuple for line color
        
    Returns:
        PIL Image with the generated grid
    """
    # Use high resolution for sub-pixel accuracy, then downsample
    scale_factor = 8  # 8x supersampling for sub-pixel accuracy
    
    # Create high-resolution image
    hr_width = width * scale_factor
    hr_height = height * scale_factor
    hr_grid_size = grid_size * scale_factor
    hr_line_width = line_width * scale_factor
    hr_offset_x = offset_x * scale_factor
    hr_offset_y = offset_y * scale_factor
    
    # Create high-resolution image
    hr_image = Image.new('RGB', (hr_width, hr_height), background_color)
    draw = ImageDraw.Draw(hr_image)
    
    # Draw vertical lines
    x = hr_offset_x
    while x < hr_width:
        # Draw line with sub-pixel width
        left = max(0, x - hr_line_width / 2)
        right = min(hr_width, x + hr_line_width / 2)
        if right > left:
            draw.rectangle([left, 0, right, hr_height], fill=line_color)
        x += hr_grid_size
    
    # Draw horizontal lines
    y = hr_offset_y
    while y < hr_height:
        # Draw line with sub-pixel width
        top = max(0, y - hr_line_width / 2)
        bottom = min(hr_height, y + hr_line_width / 2)
        if bottom > top:
            draw.rectangle([0, top, hr_width, bottom], fill=line_color)
        y += hr_grid_size
    
    # Downsample to target resolution with anti-aliasing
    final_image = hr_image.resize((width, height), Image.LANCZOS)
    
    return final_image


def create_numpy_subpixel_grid(width, height, grid_size, line_width=1.0, offset_x=0.0, offset_y=0.0):
    """
    Create a grid using numpy for maximum sub-pixel precision.
    Returns a numpy array with values between 0 and 1.
    """
    # Create coordinate grids
    x = np.linspace(0, width, width, endpoint=False)
    y = np.linspace(0, height, height, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    # Shift coordinates by offset
    X_shifted = X - offset_x
    Y_shifted = Y - offset_y
    
    # Calculate distance to nearest grid lines
    x_dist = np.abs(X_shifted % grid_size - grid_size/2) - line_width/2
    y_dist = np.abs(Y_shifted % grid_size - grid_size/2) - line_width/2
    
    # Create grid mask (1 where grid lines should be, 0 elsewhere)
    grid_mask = np.logical_or(x_dist <= 0, y_dist <= 0)
    
    # For anti-aliasing, create smooth transitions
    x_smooth = np.clip(1 - x_dist, 0, 1)
    y_smooth = np.clip(1 - y_dist, 0, 1)
    
    # Combine horizontal and vertical components
    grid_intensity = np.maximum(x_smooth, y_smooth)
    
    return grid_intensity


def main():
    parser = argparse.ArgumentParser(
        description='Generate test grid images with sub-pixel accuracy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--width', type=int, default=512,
                       help='Image width in pixels')
    parser.add_argument('--height', type=int, default=512,
                       help='Image height in pixels')
    parser.add_argument('--grid-size', type=float, required=True,
                       help='Grid spacing in pixels (can be fractional, e.g., 16.1)')
    parser.add_argument('--line-width', type=float, default=1.0,
                       help='Width of grid lines in pixels (can be fractional)')
    parser.add_argument('--offset-x', type=float, default=0.0,
                       help='Horizontal offset of grid in pixels')
    parser.add_argument('--offset-y', type=float, default=0.0,
                       help='Vertical offset of grid in pixels')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image file path')
    parser.add_argument('--method', choices=['pil', 'numpy'], default='numpy',
                       help='Rendering method: PIL with supersampling or numpy with analytical')
    parser.add_argument('--background-color', nargs=3, type=int, default=[255, 255, 255],
                       metavar=('R', 'G', 'B'), help='Background color (RGB)')
    parser.add_argument('--line-color', nargs=3, type=int, default=[0, 0, 0],
                       metavar=('R', 'G', 'B'), help='Line color (RGB)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.width <= 0 or args.height <= 0:
        print("Error: Width and height must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.grid_size <= 0:
        print("Error: Grid size must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.line_width <= 0:
        print("Error: Line width must be positive", file=sys.stderr)
        sys.exit(1)
    
    # Validate color values
    for color in [args.background_color, args.line_color]:
        if any(c < 0 or c > 255 for c in color):
            print("Error: Color values must be between 0 and 255", file=sys.stderr)
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Generating {args.width}x{args.height} grid with size {args.grid_size:.3f} pixels...")
    print(f"Line width: {args.line_width:.3f}, Offset: ({args.offset_x:.3f}, {args.offset_y:.3f})")
    print(f"Method: {args.method}")
    
    if args.method == 'pil':
        # Use PIL with supersampling
        image = create_subpixel_grid(
            args.width, args.height, args.grid_size, args.line_width,
            args.offset_x, args.offset_y,
            tuple(args.background_color), tuple(args.line_color)
        )
    else:
        # Use numpy analytical method
        grid_intensity = create_numpy_subpixel_grid(
            args.width, args.height, args.grid_size, args.line_width,
            args.offset_x, args.offset_y
        )
        
        # Convert to RGB image
        bg_color = np.array(args.background_color)
        line_color = np.array(args.line_color)
        
        # Blend colors based on grid intensity
        rgb_array = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        for c in range(3):
            rgb_array[:, :, c] = (
                bg_color[c] * (1 - grid_intensity) + 
                line_color[c] * grid_intensity
            ).astype(np.uint8)
        
        image = Image.fromarray(rgb_array)
    
    # Save the image
    image.save(args.output)
    print(f"Grid saved to: {args.output}")
    
    # Print some analysis
    grid_lines_x = int(args.width / args.grid_size) + 1
    grid_lines_y = int(args.height / args.grid_size) + 1
    print(f"Approximate grid lines: {grid_lines_x} vertical, {grid_lines_y} horizontal")
    print(f"Pixels per grid cell: {args.grid_size:.3f}")


if __name__ == '__main__':
    main()