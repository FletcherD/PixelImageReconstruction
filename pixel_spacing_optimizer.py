import argparse
import os.path
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.optimize import minimize


def load_and_convert_image(image_path):
    """Load an image and convert to greyscale numpy array with optional transforms.
    
    Args:
        image_path (str): Path to the input image file
        offset_x (int): Horizontal offset in pixels (default: 0)
        offset_y (int): Vertical offset in pixels (default: 0)
        scale_x (float): Horizontal scaling factor (default: 1.0)
        scale_y (float): Vertical scaling factor (default: 1.0)
        
    Returns:
        tuple: (PIL Image, numpy array) - Original color image and greyscale array
    """
    img = Image.open(image_path)
    
    img_grey = img.convert('L')
    img_array = np.array(img_grey, dtype=np.float32)
    return img, img_array


def detect_edges_along_axis(image, axis=0, n_deriv=1):
    """
    Apply derivative-based edge detection by taking differences between adjacent pixels
    """
    # Calculate derivatives in axis direction
    dx = np.zeros_like(image, dtype=np.float32)

    r = image
    for i in range(n_deriv):
        r = np.diff(r, axis=axis)

    if axis == 1:
        dx[:, n_deriv:] = r
    else:
        dx[n_deriv:, :] = r

    return dx

def create_intensity_profile(edge_image, axis=0):
    """Create 1D intensity profile by averaging edge image along specified axis.
    
    Args:
        edge_image (np.ndarray): Edge-detected image
        axis (int): Axis along which to average (0=average rows, 1=average columns)
        
    Returns:
        tuple: (profile array, position array) - 1D intensity profile and pixel positions
    """
    profile = np.mean(edge_image, axis=axis)
    if axis == 0:
        positions = np.arange(edge_image.shape[1])
    else:
        positions = np.arange(edge_image.shape[0])
    return profile, positions


def find_intensity_peaks(profile, positions):
    """Find significant peaks and valleys in the 1D intensity profile.
    
    Args:
        profile (np.ndarray): 1D intensity profile
        positions (np.ndarray): Pixel positions corresponding to profile values
        
    Returns:
        np.ndarray: Array of peak positions in pixels
    """
    threshold = 0.5 * np.std(profile)
    peaks_high, _ = find_peaks(profile, height=threshold, distance=1)
    peaks_low, _ = find_peaks(profile * -1.0, height=threshold, distance=1)
    peaks = np.concatenate([peaks_high, peaks_low])
    return peaks


def calculate_spacing_error(peak_positions, spacing):
    """Calculate how well a given spacing fits the detected peaks.
    
    Args:
        peak_positions (np.ndarray): Array of detected peak positions
        spacing (float): Candidate pixel spacing to test
        
    Returns:
        float: Normalized error score (lower is better)
    """
    total_error = 0
    offset = peak_positions[0]
    
    for peak in peak_positions:
        k = round((peak - offset) / spacing)
        predicted_position = offset + k * spacing
        error = (peak - predicted_position) ** 2
        total_error += error
    
    total_error = np.sqrt(total_error)
    total_error /= spacing
    return total_error


def find_candidate_spacings(peaks):
    """Find candidate pixel spacings using coarse grid search.
    
    Args:
        peaks (np.ndarray): Array of detected peak positions
        
    Returns:
        np.ndarray: Array of candidate spacing values that show good peak alignment
    """
    spacings = np.arange(2.2, 10.0, 0.01)
    spacing_errors = [calculate_spacing_error(peaks, s) for s in spacings]
    spacing_errors = np.array(spacing_errors) * -1.0
    
    threshold = np.mean(spacing_errors) + 2.0 * np.std(spacing_errors)
    error_peaks, _ = find_peaks(spacing_errors, height=threshold, distance=1)
    
    peak_spacings = spacings[error_peaks]
    return peak_spacings


def refine_spacing_estimate(peaks, initial_guess):
    """Fine-tune spacing estimate using numerical optimization.
    
    Args:
        peaks (np.ndarray): Array of detected peak positions
        initial_guess (float): Initial spacing estimate to refine
        
    Returns:
        float: Optimized pixel spacing value
    """
    def objective(params):
        spacing = params[0]
        return calculate_spacing_error(peaks, spacing)
    
    result = minimize(objective, [initial_guess], method='Nelder-Mead')
    optimal_spacing, = result.x
    return optimal_spacing


def determine_optimal_spacing(peaks):
    """Determine the optimal pixel spacing that best fits the detected peaks.
    
    Args:
        peaks (np.ndarray): Array of detected peak positions
        
    Returns:
        float: Optimal pixel spacing value
    """
    error_peaks = find_candidate_spacings(peaks)
    candidate_spacings = [refine_spacing_estimate(peaks, p) for p in error_peaks]
    candidate_errors = [calculate_spacing_error(peaks, spacing) for spacing in candidate_spacings]
    optimal_spacing = candidate_spacings[np.argmin(candidate_errors)]
    return optimal_spacing


def rescale_image_to_target_spacing(image, pixel_spacing_h, pixel_spacing_v, target_spacing,
                                     x_scale=1.0, y_scale=1.0, x_offset=0.0, y_offset=0.0):

    """Rescale image to achieve target pixel spacing.
    
    Args:
        image (PIL.Image): Input image to rescale
        pixel_spacing_h (float): Current horizontal pixel spacing
        pixel_spacing_v (float): Current vertical pixel spacing
        target_spacing (float): Desired pixel spacing after rescaling
        
    Returns:
        PIL.Image: Rescaled image
    """
      # Calculate base scaling factors
    h_scale_factor = target_spacing / pixel_spacing_h
    v_scale_factor = target_spacing / pixel_spacing_v

    # Calculate output dimensions
    output_width = int(image.size[0] * h_scale_factor * y_scale)
    output_height = int(image.size[1] * v_scale_factor* x_scale)


    return image.resize((output_width, output_height))





def optimize_pixel_spacing_and_rescale(image_path, output_path, new_spacing=4.0, offset_x=0, offset_y=0, scale_x=1.0, scale_y=1.0):
    """
    Main function to find optimal pixel spacing and rescale image
    
    Args:
        image_path: Path to input image
        output_path: Path to save rescaled image
        new_spacing: Target pixel spacing for rescaling
        offset_x: Horizontal offset in pixels for testing
        offset_y: Vertical offset in pixels for testing
        scale_x: Horizontal scaling factor for testing
        scale_y: Vertical scaling factor for testing
    """
    # Load and process image
    img_color, img_grey = load_and_convert_image(image_path)
    
    # Create edge-detected profiles
    img_edges_h = detect_edges_along_axis(img_grey, axis=1)
    h_profile, x_pos = create_intensity_profile(img_edges_h, axis=0)
    
    img_edges_v = detect_edges_along_axis(img_grey, axis=0)
    v_profile, y_pos = create_intensity_profile(img_edges_v, axis=1)
    
    # Find peaks
    h_peaks = find_intensity_peaks(h_profile, x_pos)
    v_peaks = find_intensity_peaks(v_profile, y_pos)
    print(f"Peaks found - Horizontal: {len(h_peaks)}, Vertical: {len(v_peaks)}")
    
    # Find optimal spacings
    optimal_spacing_h = determine_optimal_spacing(h_peaks)
    optimal_spacing_v = determine_optimal_spacing(v_peaks)
    print(f"Optimal spacings - Horizontal: {optimal_spacing_h:.2f}, Vertical: {optimal_spacing_v:.2f}")
    
    # Rescale and save image
    img_scaled = rescale_image_to_target_spacing(img_color, optimal_spacing_h, optimal_spacing_v, new_spacing,
                                                 scale_x, scale_y, offset_x, offset_y)
    img_scaled.save(output_path)

    print(f"Original image dimensions: {img_color.size[0]}x{img_color.size[1]}")
    print(f"New image dimensions: {img_scaled.size[0]}x{img_scaled.size[1]}")
    
    return optimal_spacing_h, optimal_spacing_v, img_scaled


def main():
    """Main entry point for the pixel spacing optimizer script."""
    parser = argparse.ArgumentParser(
        description="Prepare an image for U-Net by optimizing pixel spacing and rescaling"
    )
    parser.add_argument(
        "input", 
        help="Path to the input image file"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Path to the output image file (default: {input}_prepared.png)"
    )
    parser.add_argument(
        "-s", "--spacing", 
        type=float, 
        default=4.0,
        help="Target pixel spacing for rescaling (default: 4.0)"
    )
    parser.add_argument(
        "--offset-x",
        type=int,
        default=0,
        help="Horizontal offset in pixels for testing (default: 0)"
    )
    parser.add_argument(
        "--offset-y",
        type=int,
        default=0,
        help="Vertical offset in pixels for testing (default: 0)"
    )
    parser.add_argument(
        "--scale-x",
        type=float,
        default=1.0,
        help="Horizontal scaling factor for testing (default: 1.0)"
    )
    parser.add_argument(
        "--scale-y",
        type=float,
        default=1.0,
        help="Vertical scaling factor for testing (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        input_name, input_ext = os.path.splitext(args.input)
        args.output = f"{input_name}_prepared.png"
    
    # Process the image
    print(f"Processing image: {args.input}")
    
    spacing_h, spacing_v, img_scaled = optimize_pixel_spacing_and_rescale(
        args.input, args.output, args.spacing, args.offset_x, args.offset_y, args.scale_x, args.scale_y
    )

    print(f"Image prepared and saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
