# Pixel Image Reconstruction

This project focuses on Discrete Fourier Transform (DFT) analysis of images, particularly for edge detection and frequency domain analysis to determine pixel spacing in images.

## Project Structure

```
├── notebooks/           # Jupyter notebooks for experimentation
│   ├── edge_analysis_1d.ipynb          # Main DFT analysis workflow
│   └── color_palette_clustering.ipynb  # Color analysis experiments
├── ml-training/         # Machine learning model training
│   ├── train_pixel_unet.py            # U-Net training script
│   ├── pixel_unet.py                  # U-Net model definition
│   ├── create_training_data.py        # Dataset generation
│   ├── data_synthesis_pipeline.py     # Data synthesis pipeline
│   ├── checkpoints/                   # Model checkpoints
│   └── wandb/                         # Weights & Biases logs
├── web-app/             # WebAssembly web application
│   ├── src/                           # Rust source code
│   ├── pkg/                           # Built WASM package
│   ├── demo.html                      # Simple web demo
│   ├── example.html                   # Feature-rich demo
│   ├── Cargo.toml                     # Rust dependencies
│   └── build.sh                       # Build script
├── test-images/         # Test images and utilities
│   ├── *.jpg, *.png                   # Test images
│   └── create_checkerboard.py         # Generate test patterns
├── docs/                # Documentation
│   ├── README.md                      # Original README
│   └── README_WASM.md                 # WASM documentation
├── main.py              # Entry point script
├── pyproject.toml       # Python project configuration
├── uv.lock              # Python dependency lock file
└── CLAUDE.md            # Development guidelines
```

## Quick Start

### Notebooks (Experimentation)
```bash
uv run jupyter lab
# Open notebooks/edge_analysis_1d.ipynb
```

### ML Training
```bash
cd ml-training
uv run python train_pixel_unet.py
```

### Web Application
```bash
cd web-app
./build.sh
python -m http.server 8000
# Open http://localhost:8000/demo.html
```

## Components

### 1. Notebooks (`notebooks/`)
Interactive Jupyter notebooks for research and experimentation:
- **edge_analysis_1d.ipynb**: Main analysis pipeline with DFT, edge detection, and spacing calculation
- **color_palette_clustering.ipynb**: Color analysis and clustering experiments

### 2. ML Training (`ml-training/`)
Machine learning components for pixel reconstruction:
- U-Net model for super-resolution/reconstruction
- Training data generation and synthesis
- Model checkpoints and training logs

### 3. Web Application (`web-app/`)
WebAssembly implementation for browser-based analysis:
- Rust implementation of all notebook algorithms
- Real-time image analysis in the browser
- No server required - runs entirely client-side

### 4. Test Images (`test-images/`)
Test patterns and utilities:
- Sample images for analysis
- Checkerboard pattern generator
- Various test cases for algorithm validation

## Development Environment

**Package Manager**: UV (modern Python package manager)
- Install dependencies: `uv sync`
- Run scripts: `uv run python script.py`
- Start Jupyter: `uv run jupyter lab`

**Python Version**: Requires Python >=3.13

## Key Dependencies

- **numpy**: Core numerical operations and FFT calculations
- **opencv-python**: Canny edge detection
- **pillow**: Image loading and basic manipulation
- **plotly**: Interactive visualizations
- **scipy**: Signal processing and optimization
- **pytorch**: Machine learning framework
- **jupyter + ipywidgets**: Notebook environment

## Architecture

The project implements a complete pipeline for pixel spacing analysis:

1. **Image Preprocessing**: Convert to greyscale
2. **Edge Detection**: Second-order derivative edge detection
3. **Profile Analysis**: Create 1D profiles by averaging along axes
4. **Peak Detection**: Find peaks using adaptive thresholding
5. **Spacing Optimization**: Two-stage optimization to find optimal pixel spacing

This pipeline is implemented in three different environments:
- **Python/Jupyter**: For research and experimentation
- **PyTorch**: For ML-based approaches
- **Rust/WASM**: For high-performance web deployment

## Getting Started

1. **Clone and setup**:
   ```bash
   uv sync
   ```

2. **Start with notebooks**:
   ```bash
   uv run jupyter lab
   ```

3. **Try the web app**:
   ```bash
   cd web-app && ./build.sh
   python -m http.server 8000
   ```

4. **Train models**:
   ```bash
   cd ml-training
   uv run python train_pixel_unet.py
   ```