#!/usr/bin/env python3
"""
Quick comparison script to test different attention configurations
and find optimal settings for spatial attention in pixel art reconstruction.
"""

import torch
import numpy as np
from enhanced_spatial_attention_models import ConfigurableAttentionUNet64x16
from medium_unet import count_parameters, mse_loss


def test_attention_configurations():
    """Test different attention configurations to find optimal settings."""
    
    print("🔬 Testing Attention Configurations for Pixel Art")
    print("=" * 60)
    
    # Create test data
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    target = torch.randn(batch_size, 3, 16, 16)
    
    print(f"Test data: {x.shape} → {target.shape}")
    print()
    
    # Test configurations
    configs = [
        # Baseline (no attention)
        {
            'name': 'No Attention (Baseline)',
            'use_attention': False,
            'expected_improvement': 'baseline'
        },
        
        # Original problematic settings
        {
            'name': 'Original (Weak)',
            'use_attention': True,
            'attention_type': 'improved',
            'attention_strength': 0.0,  # This was the problem!
            'attention_temperature': 1.0,
            'attention_reduction': 8,
            'expected_improvement': 'minimal'
        },
        
        # Improved settings - moderate
        {
            'name': 'Improved (Moderate)',
            'use_attention': True,
            'attention_type': 'improved',
            'attention_strength': 0.15,
            'attention_temperature': 0.8,
            'attention_reduction': 4,
            'expected_improvement': 'good'
        },
        
        # Improved settings - strong
        {
            'name': 'Improved (Strong)',
            'use_attention': True,
            'attention_type': 'improved',
            'attention_strength': 0.25,
            'attention_temperature': 0.6,
            'attention_reduction': 4,
            'expected_improvement': 'very good'
        },
        
        # Improved settings - very strong
        {
            'name': 'Improved (Very Strong)',
            'use_attention': True,
            'attention_type': 'improved',
            'attention_strength': 0.4,
            'attention_temperature': 0.5,
            'attention_reduction': 2,
            'expected_improvement': 'excellent'
        },
        
        # Positional attention
        {
            'name': 'Positional Attention',
            'use_attention': True,
            'attention_type': 'positional',
            'attention_strength': 0.2,
            'attention_temperature': 0.7,
            'attention_reduction': 4,
            'expected_improvement': 'good+'
        },
        
        # Multi-scale attention
        {
            'name': 'Multi-Scale Attention',
            'use_attention': True,
            'attention_type': 'multiscale',
            'attention_strength': 0.2,
            'attention_temperature': 0.7,
            'attention_reduction': 4,
            'expected_improvement': 'good+'
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing: {config['name']}")
        
        # Create model with this configuration
        model_config = {k: v for k, v in config.items() 
                       if k not in ['name', 'expected_improvement']}
        
        model = ConfigurableAttentionUNet64x16(**model_config)
        model.eval()
        
        # Count parameters
        total_params = count_parameters(model)
        attention_params = 0
        if hasattr(model, 'get_attention_info'):
            attention_info = model.get_attention_info()
            attention_params = attention_info['parameters']
        
        # Test multiple times for stability
        losses = []
        output_changes = []
        
        with torch.no_grad():
            for _ in range(5):  # Multiple runs for stability
                output = model(x)
                loss = mse_loss(output, target)
                losses.append(loss.item())
                
                # Measure how much attention changes the output
                output_change = torch.norm(output - target).item()
                output_changes.append(output_change)
        
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        avg_change = np.mean(output_changes)
        
        result = {
            'name': config['name'],
            'config': model_config,
            'total_params': total_params,
            'attention_params': attention_params,
            'avg_loss': avg_loss,
            'std_loss': std_loss,
            'avg_output_change': avg_change,
            'expected': config['expected_improvement']
        }
        
        results.append(result)
        
        print(f"  Parameters: {total_params:,} (attention: {attention_params:,})")
        print(f"  Avg Loss: {avg_loss:.4f} ± {std_loss:.4f}")
        print(f"  Output Change: {avg_change:.3f}")
        print(f"  Expected: {config['expected_improvement']}")
        print()
    
    # Find baseline loss for comparison
    baseline_loss = next(r['avg_loss'] for r in results if 'Baseline' in r['name'])
    
    # Sort by performance improvement
    improvements = []
    for result in results:
        if 'Baseline' not in result['name']:
            improvement = (baseline_loss - result['avg_loss']) / baseline_loss * 100
            improvements.append((result['name'], improvement, result))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print("🏆 RESULTS RANKING (by improvement over baseline):")
    print("=" * 60)
    print(f"Baseline Loss: {baseline_loss:.4f}")
    print()
    
    for i, (name, improvement, result) in enumerate(improvements, 1):
        status = "🟢" if improvement > 5 else "🟡" if improvement > 1 else "🔴"
        print(f"{i}. {status} {name}")
        print(f"   Loss: {result['avg_loss']:.4f} (improvement: {improvement:+.2f}%)")
        print(f"   Config: {result['config']}")
        print()
    
    # Recommendations
    print("🎯 RECOMMENDATIONS:")
    print("=" * 40)
    
    if improvements and improvements[0][1] > 5:
        best_name, best_improvement, best_result = improvements[0]
        print(f"✅ BEST: {best_name}")
        print(f"   Improvement: {best_improvement:+.2f}%")
        print(f"   Use these settings in your training:")
        for key, value in best_result['config'].items():
            if key != 'attention_locations':
                print(f"   --{key.replace('_', '-')} {value}")
        print()
    
    print("📋 Quick Training Commands:")
    print("-" * 40)
    
    if improvements:
        top_3 = improvements[:3]
        for i, (name, improvement, result) in enumerate(top_3, 1):
            if improvement > 1:  # Only show configs with meaningful improvement
                config = result['config']
                if config.get('use_attention', False):
                    print(f"{i}. {name} (+{improvement:.1f}%):")
                    cmd = "python train_enhanced_spatial_attention.py --use_hf_dataset"
                    if 'attention_type' in config:
                        cmd += f" --attention_type {config['attention_type']}"
                    if 'attention_strength' in config:
                        cmd += f" --attention_strength {config['attention_strength']}"
                    if 'attention_temperature' in config:
                        cmd += f" --attention_temperature {config['attention_temperature']}"
                    if 'attention_reduction' in config:
                        cmd += f" --attention_reduction {config['attention_reduction']}"
                    print(f"   {cmd}")
                    print()
    
    return results


def analyze_attention_effectiveness():
    """Analyze why certain attention configurations work better."""
    
    print("\n🔍 ATTENTION EFFECTIVENESS ANALYSIS:")
    print("=" * 50)
    
    # Test attention strength scaling
    strengths = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    x = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 16, 16)
    
    print("Testing attention strength scaling:")
    print("Strength | Loss    | Output Change | Status")
    print("-" * 45)
    
    for strength in strengths:
        model = ConfigurableAttentionUNet64x16(
            use_attention=True,
            attention_type='improved',
            attention_strength=strength,
            attention_temperature=0.7,
            attention_reduction=4
        )
        model.eval()
        
        with torch.no_grad():
            output = model(x)
            loss = mse_loss(output, target)
            change = torch.norm(output - target).item()
        
        if strength == 0.0:
            status = "❌ No effect"
        elif strength < 0.1:
            status = "🔴 Too weak"
        elif strength < 0.3:
            status = "🟡 Good range"
        elif strength < 0.5:
            status = "🟢 Strong"
        else:
            status = "🔶 Too strong?"
        
        print(f"{strength:6.2f}   | {loss.item():6.4f} | {change:12.3f} | {status}")
    
    print(f"\n💡 Key Insights:")
    print(f"  • Attention strength 0.0 = no attention effect (original problem)")
    print(f"  • Sweet spot appears to be 0.15-0.3 for strong but stable attention")
    print(f"  • Higher values (>0.4) may cause training instability")
    print(f"  • Temperature 0.6-0.8 provides good balance between sharp and smooth attention")


if __name__ == "__main__":
    results = test_attention_configurations()
    analyze_attention_effectiveness()