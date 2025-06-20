# Project Organization Summary

The PixelImageReconstruction project has been reorganized into logical components:

## 📁 Directory Structure

```
📦 PixelImageReconstruction/
├── 📚 notebooks/           # Jupyter notebooks for experimentation
├── 🤖 ml-training/         # ML model training and synthesis
├── 🌐 web-app/             # Rust WASM web application  
├── 🖼️ test-images/         # Test images and utilities
├── 📖 docs/                # Documentation files
├── 📋 main.py              # Entry point script
├── ⚙️ pyproject.toml       # Python project config
├── 🔒 uv.lock              # Dependency lock file
└── 📝 CLAUDE.md            # Development guidelines
```

## 🚀 Quick Commands

| Component | Directory | Command |
|-----------|-----------|---------|
| **Notebooks** | `notebooks/` | `uv run jupyter lab` |
| **ML Training** | `ml-training/` | `cd ml-training && uv run python train_pixel_unet.py` |
| **Web App** | `web-app/` | `cd web-app && ./build.sh && python -m http.server 8000` |
| **Test Images** | `test-images/` | `cd test-images && uv run python create_checkerboard.py` |

## 🎯 What Each Component Does

### 📚 Notebooks
- **edge_analysis_1d.ipynb**: Core image analysis with DFT and spacing detection
- **color_palette_clustering.ipynb**: Color analysis experiments

### 🤖 ML Training
- **train_pixel_unet.py**: Train U-Net for super-resolution
- **pixel_unet.py**: U-Net model architecture
- **create_training_data.py**: Generate synthetic training data
- **data_synthesis_pipeline.py**: Complete data pipeline
- **checkpoints/**: Saved model weights
- **wandb/**: Training logs and metrics

### 🌐 Web App
- **src/lib.rs**: Rust implementation of all algorithms
- **demo.html**: Simple web interface
- **example.html**: Feature-rich demo
- **pkg/**: Built WASM package
- **Cargo.toml**: Rust dependencies

### 🖼️ Test Images
- **blueeyes.jpg, mario.png, mario_world.jpg**: Sample images
- **create_checkerboard.py**: Generate test patterns

### 📖 Documentation
- **README.md**: Original project documentation  
- **README_WASM.md**: Detailed WASM implementation guide

## 🔧 Development Workflow

1. **Experiment** in `notebooks/` with Jupyter
2. **Train models** in `ml-training/` with PyTorch
3. **Deploy** via `web-app/` with Rust/WASM
4. **Test** with images from `test-images/`

Each component is self-contained and can be developed independently!