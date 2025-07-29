#!/usr/bin/env python3
"""
Test file to verify model implementations work correctly.
Run this before training to ensure models are properly implemented.
"""

import torch
import numpy as np
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner


def test_mlp_planner():
    """Test MLP Planner with sample data."""
    print("Testing MLP Planner...")
    
    model = MLPPlanner()
    batch_size = 4
    
    # Sample data
    track_left = torch.randn(batch_size, 10, 2)  # (B, n_track, 2)
    track_right = torch.randn(batch_size, 10, 2)  # (B, n_track, 2)
    
    # Forward pass
    output = model(track_left=track_left, track_right=track_right)
    
    # Check output shape
    expected_shape = (batch_size, 3, 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✅ MLP Planner: Output shape {output.shape}")
    return output


def test_transformer_planner():
    """Test Transformer Planner with sample data."""
    print("Testing Transformer Planner...")
    
    model = TransformerPlanner()
    batch_size = 4
    
    # Sample data
    track_left = torch.randn(batch_size, 10, 2)
    track_right = torch.randn(batch_size, 10, 2)
    
    # Forward pass
    output = model(track_left=track_left, track_right=track_right)
    
    # Check output shape
    expected_shape = (batch_size, 3, 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✅ Transformer Planner: Output shape {output.shape}")
    return output


def test_cnn_planner():
    """Test CNN Planner with sample data."""
    print("Testing CNN Planner...")
    
    model = CNNPlanner()
    batch_size = 4
    
    # Sample data (normalized images)
    images = torch.rand(batch_size, 3, 96, 128)  # (B, 3, H, W) in [0, 1]
    
    # Forward pass
    output = model(images)
    
    # Check output shape
    expected_shape = (batch_size, 3, 2)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✅ CNN Planner: Output shape {output.shape}")
    return output


def test_model_parameters():
    """Test that models have reasonable number of parameters."""
    print("\nTesting model parameters...")
    
    models = {
        "MLP Planner": MLPPlanner(),
        "Transformer Planner": TransformerPlanner(),
        "CNN Planner": CNNPlanner()
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        print()


def main():
    """Run all tests."""
    print("🧪 Running Model Tests")
    print("=" * 50)
    
    try:
        test_mlp_planner()
        test_transformer_planner()
        test_cnn_planner()
        test_model_parameters()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 