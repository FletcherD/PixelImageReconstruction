#!/usr/bin/env python3
"""
Export PyTorch PixelUNet model to ONNX format with optional quantization.
"""

import argparse
import torch
import torch.onnx
from pathlib import Path
import sys
import os

# Import the model architecture
from pixel_unet import PixelUNet, PixelUNetGAP


def load_model(checkpoint_path: str, model_type: str = 'standard', device: str = 'cpu'):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model instance
    if model_type == 'gap':
        model = PixelUNetGAP()
    else:
        model = PixelUNet()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    return model


def export_to_onnx(model, output_path: str, input_shape: tuple = (1, 3, 128, 128), 
                   quantize: bool = False, opset_version: int = 11):
    """Export PyTorch model to ONNX format."""
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    if next(model.parameters()).is_cuda:
        dummy_input = dummy_input.cuda()
    
    # Define input and output names
    input_names = ['input']
    output_names = ['output']
    
    # Dynamic axes for variable batch size
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    print(f"Exporting model to {output_path}")
    print(f"Input shape: {input_shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True
    )
    
    # Apply quantization if requested
    if quantize:
        quantize_onnx_model(output_path)
    
    print(f"Successfully exported model to {output_path}")


def preprocess_onnx_model(onnx_path: str):
    """Preprocess ONNX model for better quantization."""
    try:
        from onnxruntime.quantization.preprocess import preprocess_model
        
        preprocessed_path = onnx_path.replace('.onnx', '_preprocessed.onnx')
        
        print(f"Preprocessing ONNX model for quantization...")
        preprocess_model(
            input_model_path=onnx_path,
            output_model_path=preprocessed_path,
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            verbose=1
        )
        
        # Replace original with preprocessed version
        os.replace(preprocessed_path, onnx_path)
        print(f"Model preprocessed successfully")
        
    except ImportError:
        print("Warning: onnxruntime preprocessing not available")
    except Exception as e:
        print(f"Preprocessing failed: {e}")


def quantize_onnx_model(onnx_path: str):
    """Apply quantization to ONNX model."""
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # First preprocess the model
        preprocess_onnx_model(onnx_path)
        
        quantized_path = onnx_path.replace('.onnx', '_quantized.onnx')
        
        print(f"Applying dynamic quantization...")
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        
        # Replace original with quantized version
        os.replace(quantized_path, onnx_path)
        print(f"Model quantized successfully")
        
    except ImportError:
        print("Warning: onnxruntime not available, skipping quantization")
        print("Install with: pip install onnxruntime")
    except Exception as e:
        print(f"Quantization failed: {e}")


def verify_export(onnx_path: str, original_model, input_shape: tuple = (1, 3, 128, 128)):
    """Verify ONNX export by comparing outputs."""
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Create test input
        test_input = torch.randn(*input_shape)
        if next(original_model.parameters()).is_cuda:
            test_input_gpu = test_input.cuda()
            with torch.no_grad():
                pytorch_output = original_model(test_input_gpu).cpu()
        else:
            with torch.no_grad():
                pytorch_output = original_model(test_input)
        
        # Run ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output.numpy() - onnx_output))
        print(f"Max difference between PyTorch and ONNX outputs: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✓ Export verification successful!")
        else:
            print("⚠ Large difference detected, please check the export")
            
    except ImportError:
        print("onnxruntime not available, skipping verification")
    except Exception as e:
        print(f"Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch PixelUNet model to ONNX')
    parser.add_argument('--checkpoint', '-c', type=str, 
                       default='checkpoints-33-128/final_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', '-o', type=str,
                       help='Output ONNX file path (default: based on checkpoint name)')
    parser.add_argument('--model-type', choices=['standard', 'gap'], default='standard',
                       help='Model architecture type')
    parser.add_argument('--quantize', '-q', action='store_true',
                       help='Apply quantization to reduce model size')
    parser.add_argument('--input-size', type=int, nargs=2, default=[128, 128],
                       help='Input image size (height width)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use for export')
    parser.add_argument('--verify', action='store_true',
                       help='Verify export by comparing outputs')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file {args.checkpoint} not found")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem
        args.output = f"{checkpoint_name}.onnx"
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    try:
        # Load model
        model = load_model(args.checkpoint, args.model_type, device)
        
        # Define input shape
        input_shape = (args.batch_size, 3, args.input_size[0], args.input_size[1])
        
        # Export to ONNX
        export_to_onnx(
            model, 
            args.output, 
            input_shape, 
            args.quantize, 
            args.opset_version
        )
        
        # Verify export if requested
        if args.verify:
            verify_export(args.output, model, input_shape)
        
        # Print file size
        file_size = Path(args.output).stat().st_size / (1024 * 1024)
        print(f"Exported model size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()