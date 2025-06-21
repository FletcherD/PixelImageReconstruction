import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.optimize import minimize


def load_and_convert_image(image_path):
    """Load an image and convert to greyscale numpy array"""
    img = Image.open(image_path)
    img_grey = img.convert('L')
    img_array = np.array(img_grey, dtype=np.float32)
    return img, img_array


def apply_derivative_edge_detection_along_axis(image, axis=0):
    """Apply derivative-based edge detection along specified axis"""
    dx = np.zeros_like(image, dtype=np.float32)
    if axis == 1:
        dx[:, 2:] = np.diff(np.diff(image, axis=axis), axis=axis)
    else:
        dx[2:, :] = np.diff(np.diff(image, axis=axis), axis=axis)
    return dx


def create_1d_profile(edge_image, axis=0):
    """Create 1D profile by averaging edge image along specified axis"""
    profile = np.mean(edge_image, axis=axis)
    if axis == 0:
        positions = np.arange(edge_image.shape[1])
    else:
        positions = np.arange(edge_image.shape[0])
    return profile, positions


def find_profile_peaks(profile, positions):
    """Find peaks in the 1D profile"""
    threshold = 0.5 * np.std(profile)
    peaks_high, _ = find_peaks(profile, height=threshold, distance=1)
    peaks_low, _ = find_peaks(profile * -1.0, height=threshold, distance=1)
    peaks = np.concatenate([peaks_high, peaks_low])
    return peaks


def get_peaks_error(peak_positions, spacing):
    """Calculate error for a given spacing"""
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


def find_minimum_coarse(peaks):
    """Find coarse estimate of optimal spacing"""
    spacings = np.arange(2.0, 10.0, 0.01)
    spacing_errors = [get_peaks_error(peaks, s) for s in spacings]
    spacing_errors = np.array(spacing_errors) * -1.0
    
    threshold = np.mean(spacing_errors) + 5.0 * np.std(spacing_errors)
    error_peaks, _ = find_peaks(spacing_errors, height=threshold, distance=1)
    
    peak_spacings = spacings[error_peaks]
    return peak_spacings


def find_minimum_fine(peaks, initial_guess):
    """Fine-tune spacing estimate using optimization"""
    def objective(params):
        spacing = params[0]
        return get_peaks_error(peaks, spacing)
    
    result = minimize(objective, [initial_guess], method='Nelder-Mead')
    optimal_spacing, = result.x
    return optimal_spacing


def find_optimal_spacing(peaks):
    """Find optimal spacing for given peaks"""
    error_peaks = find_minimum_coarse(peaks)
    candidate_spacings = [find_minimum_fine(peaks, p) for p in error_peaks]
    candidate_errors = [get_peaks_error(peaks, spacing) for spacing in candidate_spacings]
    optimal_spacing = candidate_spacings[np.argmin(candidate_errors)]
    return optimal_spacing


def rescale_image(image, pixel_spacing_h, pixel_spacing_v, new_spacing):
    """Rescale image based on pixel spacings"""
    h_scale = new_spacing / pixel_spacing_h
    v_scale = new_spacing / pixel_spacing_v
    return image.resize((int(image.size[0] * h_scale), int(image.size[1] * v_scale)))


def optimize_pixel_spacing_and_rescale(image_path, output_path, new_spacing=4.0):
    """
    Main function to find optimal pixel spacing and rescale image
    
    Args:
        image_path: Path to input image
        output_path: Path to save rescaled image
        new_spacing: Target pixel spacing for rescaling
    """
    # Load and process image
    img_color, img_grey = load_and_convert_image(image_path)
    
    # Create edge-detected profiles
    img_edges_h = apply_derivative_edge_detection_along_axis(img_grey, axis=1)
    h_profile, x_pos = create_1d_profile(img_edges_h, axis=0)
    
    img_edges_v = apply_derivative_edge_detection_along_axis(img_grey, axis=0)
    v_profile, y_pos = create_1d_profile(img_edges_v, axis=1)
    
    # Find peaks
    h_peaks = find_profile_peaks(h_profile, x_pos)
    v_peaks = find_profile_peaks(v_profile, y_pos)
    
    # Find optimal spacings
    optimal_spacing_h = find_optimal_spacing(h_peaks)
    optimal_spacing_v = find_optimal_spacing(v_peaks)
    
    # Rescale and save image
    img_scaled = rescale_image(img_color, optimal_spacing_h, optimal_spacing_v, new_spacing)
    img_scaled.save(output_path)
    
    return optimal_spacing_h, optimal_spacing_v, img_scaled