use wasm_bindgen::prelude::*;
use std::f64::consts::PI;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Complete image analysis pipeline - takes image data and returns optimal spacing
#[wasm_bindgen]
pub fn analyze_image_spacing(data: &[u8], width: usize, height: usize) -> Vec<f64> {
    console_error_panic_hook::set_once();
    
    console_log!("=== Starting image analysis: {}x{} ===", width, height);
    
    // Convert RGBA to greyscale
    let grey_image = convert_rgba_to_greyscale(data, width, height);
    let grey_stats = get_image_stats(&grey_image);
    console_log!("Greyscale conversion - min: {:.2}, max: {:.2}, mean: {:.2}", 
                grey_stats.0, grey_stats.1, grey_stats.2);
    
    // Apply edge detection along each axis
    let edges_h = apply_derivative_edge_detection_along_axis(&grey_image, width, height, 1);
    let edges_v = apply_derivative_edge_detection_along_axis(&grey_image, width, height, 0);
    
    let edge_h_stats = get_image_stats(&edges_h);
    let edge_v_stats = get_image_stats(&edges_v);
    console_log!("Horizontal edges - min: {:.2}, max: {:.2}, mean: {:.2}", 
                edge_h_stats.0, edge_h_stats.1, edge_h_stats.2);
    console_log!("Vertical edges - min: {:.2}, max: {:.2}, mean: {:.2}", 
                edge_v_stats.0, edge_v_stats.1, edge_v_stats.2);
    
    // Create 1D profiles
    let h_profile = create_1d_profile(&edges_h, width, height, 0);
    let v_profile = create_1d_profile(&edges_v, width, height, 1);
    
    let h_profile_stats = get_profile_stats(&h_profile);
    let v_profile_stats = get_profile_stats(&v_profile);
    console_log!("H-profile (len={}): min={:.3}, max={:.3}, mean={:.3}, std={:.3}", 
                h_profile.len(), h_profile_stats.0, h_profile_stats.1, h_profile_stats.2, h_profile_stats.3);
    console_log!("V-profile (len={}): min={:.3}, max={:.3}, mean={:.3}, std={:.3}", 
                v_profile.len(), v_profile_stats.0, v_profile_stats.1, v_profile_stats.2, v_profile_stats.3);
    
    // Find peaks
    let h_peaks = find_profile_peaks(&h_profile);
    let v_peaks = find_profile_peaks(&v_profile);
    console_log!("H-peaks ({}): {:?}", h_peaks.len(), 
                if h_peaks.len() <= 10 { format!("{:?}", h_peaks) } else { format!("{:?}...", &h_peaks[0..10]) });
    console_log!("V-peaks ({}): {:?}", v_peaks.len(), 
                if v_peaks.len() <= 10 { format!("{:?}", v_peaks) } else { format!("{:?}...", &v_peaks[0..10]) });
    
    // Find optimal spacing
    let h_spacing = if !h_peaks.is_empty() {
        let spacing = find_optimal_spacing(&h_peaks).unwrap_or(-1.0);
        console_log!("H-spacing optimization result: {:.3}", spacing);
        spacing
    } else {
        console_log!("H-spacing: no peaks found");
        -1.0
    };
    
    let v_spacing = if !v_peaks.is_empty() {
        let spacing = find_optimal_spacing(&v_peaks).unwrap_or(-1.0);
        console_log!("V-spacing optimization result: {:.3}", spacing);
        spacing
    } else {
        console_log!("V-spacing: no peaks found");
        -1.0
    };
    
    console_log!("=== Final results: H={:.3}, V={:.3} ===", h_spacing, v_spacing);
    
    vec![h_spacing, v_spacing]
}

/// Convert RGBA image data to greyscale
fn convert_rgba_to_greyscale(rgba_data: &[u8], width: usize, height: usize) -> Vec<f32> {
    let mut grey_array = Vec::with_capacity(width * height);
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let r = rgba_data[idx] as f32;
            let g = rgba_data[idx + 1] as f32;
            let b = rgba_data[idx + 2] as f32;
            
            // Use luminance formula for greyscale conversion
            let grey = 0.299 * r + 0.587 * g + 0.114 * b;
            grey_array.push(grey);
        }
    }
    
    grey_array
}

/// Apply derivative-based edge detection along specified axis
fn apply_derivative_edge_detection_along_axis(image: &[f32], width: usize, height: usize, axis: usize) -> Vec<f32> {
    let mut result = vec![0.0; width * height];
    
    if axis == 1 {
        // Horizontal derivatives (along x-axis)
        for y in 0..height {
            for x in 2..width {
                let idx = y * width + x;
                let idx_1 = y * width + (x - 1);
                let idx_2 = y * width + (x - 2);
                // Second derivative: f(x) - 2*f(x-1) + f(x-2)
                result[idx] = image[idx] - 2.0 * image[idx_1] + image[idx_2];
            }
        }
    } else {
        // Vertical derivatives (along y-axis)
        for y in 2..height {
            for x in 0..width {
                let idx = y * width + x;
                let idx_1 = (y - 1) * width + x;
                let idx_2 = (y - 2) * width + x;
                // Second derivative: f(y) - 2*f(y-1) + f(y-2)
                result[idx] = image[idx] - 2.0 * image[idx_1] + image[idx_2];
            }
        }
    }
    
    result
}

/// Create 1D profile by averaging along specified axis
fn create_1d_profile(edge_image: &[f32], width: usize, height: usize, axis: usize) -> Vec<f32> {
    if axis == 0 {
        // Average along Y-axis to get horizontal profile
        let mut profile = vec![0.0; width];
        for x in 0..width {
            let mut sum = 0.0;
            for y in 0..height {
                sum += edge_image[y * width + x];
            }
            profile[x] = sum / height as f32;
        }
        profile
    } else {
        // Average along X-axis to get vertical profile
        let mut profile = vec![0.0; height];
        for y in 0..height {
            let mut sum = 0.0;
            for x in 0..width {
                sum += edge_image[y * width + x];
            }
            profile[y] = sum / width as f32;
        }
        profile
    }
}

/// Find peaks in a 1D profile
fn find_profile_peaks(profile: &[f32]) -> Vec<f64> {
    if profile.len() < 3 {
        console_log!("Profile too short for peak detection: {}", profile.len());
        return Vec::new();
    }
    
    let mut peaks = Vec::new();
    
    // Calculate adaptive threshold
    let mean = profile.iter().sum::<f32>() / profile.len() as f32;
    let variance = profile.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / profile.len() as f32;
    let std_dev = variance.sqrt();
    let threshold = 0.5 * std_dev;
    
    console_log!("Peak detection - threshold: {:.4}, mean: {:.4}, std: {:.4}", threshold, mean, std_dev);
    
    let mut positive_peaks = 0;
    let mut negative_peaks = 0;
    
    // Find peaks (both positive and negative)
    for i in 1..profile.len()-1 {
        let current = profile[i];
        let prev = profile[i-1];
        let next = profile[i+1];
        
        // Local maximum above threshold
        if current > prev && current > next && current > threshold {
            peaks.push(i as f64);
            positive_peaks += 1;
        }
        
        // Local minimum below -threshold
        if current < prev && current < next && current < -threshold {
            peaks.push(i as f64);
            negative_peaks += 1;
        }
    }
    
    console_log!("Found {} positive peaks, {} negative peaks", positive_peaks, negative_peaks);
    
    peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    peaks
}

/// Calculate error for peak spacing
fn get_peaks_error(peak_positions: &[f64], spacing: f64) -> f64 {
    if peak_positions.is_empty() || spacing <= 0.0 {
        return f64::INFINITY;
    }
    
    let mut total_error = 0.0;
    let offset = peak_positions[0];
    
    for &peak in peak_positions {
        let k = ((peak - offset) / spacing).round();
        let predicted_position = offset + k * spacing;
        let error = (peak - predicted_position).powi(2);
        total_error += error;
    }
    
    total_error = total_error.sqrt();
    total_error / spacing // Reward larger spacings
}

/// Find coarse minima - equivalent to notebook's find_minimum_coarse
fn find_minimum_coarse(peaks: &[f64]) -> Vec<f64> {
    if peaks.is_empty() {
        return Vec::new();
    }
    
    console_log!("Finding coarse minima for {} peaks", peaks.len());
    
    // Create spacing array: np.arange(2.0, 10.0, 0.01)
    let mut spacings = Vec::new();
    let mut spacing = 2.0;
    while spacing < 10.0 {
        spacings.push(spacing);
        spacing += 0.01;
    }
    
    // Calculate spacing errors and negate them (like notebook: * -1.0)
    let spacing_errors: Vec<f64> = spacings.iter()
        .map(|&s| -get_peaks_error(peaks, s))
        .collect();
    
    console_log!("Calculated {} spacing errors", spacing_errors.len());
    
    // Calculate threshold: mean + 5.0 * std
    let mean = spacing_errors.iter().sum::<f64>() / spacing_errors.len() as f64;
    let variance = spacing_errors.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / spacing_errors.len() as f64;
    let std_dev = variance.sqrt();
    let threshold = mean + 5.0 * std_dev;
    
    console_log!("Coarse search threshold: {:.6}, mean: {:.6}, std: {:.6}", threshold, mean, std_dev);
    
    // Find peaks in the negated error function (equivalent to scipy.signal.find_peaks)
    let mut peak_spacings = Vec::new();
    
    for i in 1..spacing_errors.len()-1 {
        let current = spacing_errors[i];
        let prev = spacing_errors[i-1];
        let next = spacing_errors[i+1];
        
        // Local maximum above threshold (distance=1 is implicit)
        if current > prev && current > next && current > threshold {
            peak_spacings.push(spacings[i]);
        }
    }
    
    console_log!("Found {} coarse candidates: {:?}", peak_spacings.len(), 
                if peak_spacings.len() <= 5 { format!("{:?}", peak_spacings) } 
                else { format!("{:?}...", &peak_spacings[0..5]) });
    
    peak_spacings
}

/// Fine optimization using simple grid refinement (approximating Nelder-Mead)
fn find_minimum_fine(peaks: &[f64], initial_guess: f64) -> f64 {
    console_log!("Fine optimization starting from {:.6}", initial_guess);
    
    let mut best_spacing = initial_guess;
    let mut best_error = get_peaks_error(peaks, initial_guess);
    
    // Multiple refinement passes with decreasing step sizes
    let step_sizes = [0.1, 0.01, 0.001];
    
    for &step_size in &step_sizes {
        let search_range = step_size * 10.0;
        let start = (best_spacing - search_range).max(1.0);
        let end = best_spacing + search_range;
        
        let mut spacing = start;
        while spacing <= end {
            let error = get_peaks_error(peaks, spacing);
            if error < best_error {
                best_error = error;
                best_spacing = spacing;
            }
            spacing += step_size;
        }
    }
    
    console_log!("Fine optimization result: {:.6} (error: {:.6})", best_spacing, best_error);
    best_spacing
}

/// Find optimal spacing - matches notebook's find_optimal_spacing exactly
fn find_optimal_spacing(peaks: &[f64]) -> Option<f64> {
    if peaks.is_empty() {
        return None;
    }
    
    console_log!("Starting optimal spacing search for {} peaks", peaks.len());
    
    // Step 1: Find coarse candidates
    let error_peaks = find_minimum_coarse(peaks);
    if error_peaks.is_empty() {
        console_log!("No coarse candidates found");
        return None;
    }
    
    // Step 2: Fine optimization for each candidate
    let candidate_spacings: Vec<f64> = error_peaks.iter()
        .map(|&initial_guess| find_minimum_fine(peaks, initial_guess))
        .collect();
    
    // Step 3: Calculate errors for all candidates
    let candidate_errors: Vec<f64> = candidate_spacings.iter()
        .map(|&spacing| get_peaks_error(peaks, spacing))
        .collect();
    
    console_log!("Candidate spacings: {:?}", candidate_spacings);
    console_log!("Candidate errors: {:?}", candidate_errors);
    
    // Step 4: Find minimum error candidate
    let min_error_idx = candidate_errors.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)?;
    
    let optimal_spacing = candidate_spacings[min_error_idx];
    let optimal_error = candidate_errors[min_error_idx];
    
    console_log!("Optimal spacing: {:.6}, error: {:.6}", optimal_spacing, optimal_error);
    
    Some(optimal_spacing)
}

/// Get basic statistics for debugging
fn get_image_stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    
    (min, max, mean)
}

/// Get profile statistics including standard deviation
fn get_profile_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();
    
    (min, max, mean, std_dev)
}