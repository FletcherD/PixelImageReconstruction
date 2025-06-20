# Pixel Image Reconstruction - Rust WASM Implementation

This directory contains a Rust implementation of the image processing functions from the Jupyter notebook `edge_analysis_1d.ipynb`, compiled to WebAssembly for use in web applications.

## Overview

The Rust implementation provides the following functionality:

1. **Image Processing**: Convert RGBA/RGB images to grayscale
2. **Edge Detection**: Derivative-based edge detection using second differences
3. **Profile Analysis**: Create 1D profiles by averaging edge data along axes
4. **Peak Finding**: Detect peaks in 1D profiles with threshold-based filtering
5. **Spacing Optimization**: Find optimal pixel spacing using numerical optimization

## Project Structure

```
src/
├── lib.rs              # Main WASM interface and ImageAnalysis struct
├── utils.rs            # Utility functions and console logging
├── image_processing.rs # Image conversion functions
├── edge_detection.rs   # Derivative-based edge detection
├── profile_analysis.rs # 1D profile creation and analysis
├── peak_finding.rs     # Peak detection algorithms
└── optimization.rs     # Spacing optimization with Nelder-Mead
```

## Dependencies

The implementation uses the following Rust crates:

- **wasm-bindgen**: WebAssembly bindings for JavaScript interop
- **ndarray**: N-dimensional arrays (equivalent to NumPy)
- **ndarray-stats**: Statistical functions for arrays
- **argmin**: Optimization algorithms (Nelder-Mead solver)
- **image**: Image processing utilities
- **web-sys/js-sys**: Web API bindings

## Building

### Prerequisites

1. Install Rust: https://rustup.rs/
2. Install wasm-pack: `cargo install wasm-pack`

### Build Commands

```bash
# Build the WASM package
./build.sh

# Or manually:
wasm-pack build --target web --out-dir pkg
```

This generates a `pkg/` directory containing:
- `pixel_image_reconstruction.js` - JavaScript bindings
- `pixel_image_reconstruction_bg.wasm` - WebAssembly binary
- `pixel_image_reconstruction.d.ts` - TypeScript definitions

## Usage

### Web Application Integration

```javascript
import init, { ImageAnalysis } from './pkg/pixel_image_reconstruction.js';

async function main() {
    // Initialize WASM module
    await init();
    
    // Create analysis instance
    const analysis = new ImageAnalysis();
    
    // Load image data (RGBA format)
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    analysis.load_image_data(imageData.data, canvas.width, canvas.height);
    
    // Run analysis pipeline
    analysis.apply_edge_detection();
    analysis.create_profiles();
    analysis.find_peaks();
    analysis.calculate_optimal_spacing();
    
    // Get results
    const hSpacing = analysis.get_horizontal_spacing();
    const vSpacing = analysis.get_vertical_spacing();
    
    console.log(`Horizontal spacing: ${hSpacing} pixels`);
    console.log(`Vertical spacing: ${vSpacing} pixels`);
}
```

### API Reference

#### ImageAnalysis Class

**Constructor**
- `new ImageAnalysis()` - Create a new analysis instance

**Data Loading**
- `load_image_data(data: Uint8Array, width: number, height: number)` - Load RGBA image data

**Analysis Steps**
- `apply_edge_detection()` - Apply derivative-based edge detection
- `create_profiles()` - Create 1D profiles from edge data
- `find_peaks()` - Find peaks in the profiles
- `calculate_optimal_spacing()` - Calculate optimal pixel spacing

**Data Access**
- `get_dimensions()` - Get image dimensions [width, height]
- `get_edge_data()` - Get edge detection result as flat array
- `get_horizontal_profile()` - Get horizontal 1D profile
- `get_vertical_profile()` - Get vertical 1D profile
- `get_horizontal_peaks()` - Get horizontal peak positions
- `get_vertical_peaks()` - Get vertical peak positions
- `get_horizontal_spacing()` - Get optimal horizontal spacing
- `get_vertical_spacing()` - Get optimal vertical spacing

## Algorithm Details

### Edge Detection
Implements second-order derivative edge detection:
```rust
// For horizontal edges (axis=1)
result[y][x] = image[y][x] - 2*image[y][x-1] + image[y][x-2]

// For vertical edges (axis=0)  
result[y][x] = image[y][x] - 2*image[y-1][x] + image[y-2][x]
```

### Peak Finding
Uses adaptive thresholding based on profile statistics:
- Threshold = 0.5 × standard_deviation
- Finds both positive peaks (local maxima) and negative peaks (local minima)
- Applies minimum distance constraint between peaks

### Spacing Optimization
Two-stage optimization process:
1. **Coarse Search**: Evaluate spacing candidates from 2.0 to 10.0 pixels
2. **Fine Search**: Nelder-Mead optimization starting from coarse candidates

Error function rewards larger spacings to avoid overfitting:
```rust
error = sqrt(sum((peak - predicted_position)²)) / spacing
```

## Demo

The `example.html` file provides a complete web application demonstrating all functionality:

1. Load an image file
2. Apply edge detection and visualize results
3. Create and plot 1D profiles
4. Find peaks in the profiles
5. Calculate optimal pixel spacing

To run the demo:
1. Build the WASM package: `./build.sh`
2. Serve the files over HTTP (required for WASM):
   ```bash
   python -m http.server 8000
   # or
   npx serve .
   ```
3. Open `http://localhost:8000/example.html`

## Performance

The Rust implementation is optimized for performance:
- Uses efficient ndarray operations
- Minimal memory allocations
- Compiled with optimization level "s" for size
- WebAssembly provides near-native performance in browsers

## Testing

Run the test suite:
```bash
cargo test
```

Tests cover:
- Image conversion accuracy
- Edge detection correctness  
- Profile creation validation
- Peak finding algorithms
- Optimization convergence

## Comparison with Python

The Rust implementation maintains high fidelity to the original Python notebook:

| Function | Python | Rust Equivalent |
|----------|--------|-----------------|
| `PIL.Image.convert('L')` | `convert_rgba_to_greyscale()` |
| `np.diff(np.diff(...))` | `apply_derivative_edge_detection_along_axis()` |
| `np.mean(axis=...)` | `create_1d_profile()` |
| `scipy.signal.find_peaks` | `find_profile_peaks()` |
| `scipy.optimize.minimize` | `find_minimum_fine()` (Nelder-Mead) |

Key differences:
- Uses Rust's type safety and memory management
- Optimized for WebAssembly target
- Provides JavaScript-friendly API
- Includes comprehensive error handling