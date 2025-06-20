#!/usr/bin/env python3
"""
Example script to create training data for pixel art recovery model.
"""

import os
import torch
from torch.utils.data import DataLoader
from data_synthesis_pipeline import create_dataset, PixelArtDataSynthesizer


def main():
    # Configuration
    SOURCE_IMAGES_DIR = "pixel_art_dataset"  # Directory with your source images
    NUM_SAMPLES = 1000  # Number of training samples to generate
    BATCH_SIZE = 16
    EXAMPLES_DIR = "training_examples"
    
    print("Creating pixel art recovery dataset...")
    
    # Create the dataset
    try:
        dataset = create_dataset(
            source_images_dir=SOURCE_IMAGES_DIR,
            num_samples=NUM_SAMPLES,
            save_examples=True,
            examples_dir=EXAMPLES_DIR
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Create DataLoader for training
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=4
        )
        
        print(f"✓ DataLoader created with batch size {BATCH_SIZE}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Input batch shape: {inputs.shape}")    # [batch_size, 3, 256, 256]
            print(f"  Target batch shape: {targets.shape}")  # [batch_size, 3, 32, 32]
            
            if batch_idx >= 2:  # Just test first few batches
                break
        
        print(f"\n✓ Dataset is ready for training!")
        print(f"✓ Example images saved to: {EXAMPLES_DIR}/")
        print(f"✓ Use the DataLoader in your training loop")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print(f"Make sure the source directory '{SOURCE_IMAGES_DIR}' exists and contains images")


def test_individual_synthesis():
    """Test the synthesis pipeline on individual images."""
    print("\nTesting individual image synthesis...")
    
    # You can test with existing images in your project
    test_images = ["mario.png", "mario_world.jpg", "blueeyes.jpg"]
    
    synthesizer = PixelArtDataSynthesizer(seed=42)
    
    for img_name in test_images:
        if os.path.exists(img_name):
            from PIL import Image
            
            print(f"Processing {img_name}...")
            source_image = Image.open(img_name).convert('RGB')
            
            # Create a synthetic pair
            input_img, target_img = synthesizer.synthesize_pair(source_image)
            
            # Save results
            base_name = os.path.splitext(img_name)[0]
            input_img.save(f"test_{base_name}_input.png")
            target_img.save(f"test_{base_name}_target.png")
            
            # Save upscaled target for comparison
            target_upscaled = target_img.resize((256, 256), Image.NEAREST)
            target_upscaled.save(f"test_{base_name}_target_upscaled.png")
            
            print(f"  ✓ Saved test_{base_name}_*.png")


if __name__ == "__main__":
    main()
    test_individual_synthesis()