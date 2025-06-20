#!/bin/bash

# Build the WASM package
echo "Building WASM package..."
wasm-pack build --target web --out-dir pkg

echo "WASM package built successfully!"
echo "Generated files are in the 'pkg' directory"
echo ""
echo "To use in a web application:"
echo "1. Copy the generated pkg/ directory to your web server"
echo "2. Import the module in your JavaScript:"
echo "   import init, { ImageAnalysis } from './pkg/pixel_image_reconstruction.js';"
echo "3. Initialize and use the functions as shown in example.html"