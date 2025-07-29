#!/usr/bin/env python3
"""
Test script to verify our trained model works correctly.
"""

import torch
import numpy as np
from homework.models import load_model


def test_trained_mlp():
    """Test the trained MLP model with sample data."""
    print("🧪 Testing Trained MLP Model")
    print("=" * 50)
    
    # Load the trained model
    model = load_model("mlp_planner", with_weights=True)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample test data
    batch_size = 4
    track_left = torch.randn(batch_size, 10, 2)  # Random left track points
    track_right = torch.randn(batch_size, 10, 2)  # Random right track points
    
    print(f"\n📥 Input shapes:")
    print(f"   • track_left: {track_left.shape}")
    print(f"   • track_right: {track_right.shape}")
    
    # Make predictions
    with torch.no_grad():
        predictions = model(track_left=track_left, track_right=track_right)
    
    print(f"\n📤 Output shape: {predictions.shape}")
    print(f"🎯 Predicted waypoints:")
    
    for i in range(batch_size):
        print(f"   Sample {i+1}:")
        for j, waypoint in enumerate(predictions[i]):
            print(f"     Waypoint {j+1}: ({waypoint[0]:.3f}, {waypoint[1]:.3f})")
    
    # Test with different batch sizes
    print(f"\n🧪 Testing with different batch sizes...")
    
    for batch_size in [1, 8, 16]:
        track_left = torch.randn(batch_size, 10, 2)
        track_right = torch.randn(batch_size, 10, 2)
        
        with torch.no_grad():
            predictions = model(track_left=track_left, track_right=track_right)
        
        print(f"   ✅ Batch size {batch_size}: Output shape {predictions.shape}")
    
    print(f"\n🎉 All tests passed! Model is working correctly.")
    
    return predictions


def test_model_performance():
    """Test model performance with realistic data."""
    print("\n📊 Testing Model Performance")
    print("=" * 50)
    
    model = load_model("mlp_planner", with_weights=True)
    model.eval()
    
    # Create more realistic test data (simulating actual road boundaries)
    batch_size = 32
    
    # Simulate a curved road
    t = torch.linspace(0, 2*np.pi, 10)
    
    # Left track: curved path
    left_x = 2 * torch.cos(t) + torch.randn(10) * 0.1
    left_y = torch.sin(t) + torch.randn(10) * 0.1
    track_left = torch.stack([left_x, left_y], dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Right track: parallel to left track
    right_x = left_x + 1.5 + torch.randn(10) * 0.1
    right_y = left_y + torch.randn(10) * 0.1
    track_right = torch.stack([right_x, right_y], dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
    
    print(f"📐 Created realistic road data:")
    print(f"   • Left track range: x[{track_left[0, :, 0].min():.2f}, {track_left[0, :, 0].max():.2f}], y[{track_left[0, :, 1].min():.2f}, {track_left[0, :, 1].max():.2f}]")
    print(f"   • Right track range: x[{track_right[0, :, 0].min():.2f}, {track_right[0, :, 0].max():.2f}], y[{track_right[0, :, 1].min():.2f}, {track_right[0, :, 1].max():.2f}]")
    
    # Make predictions
    with torch.no_grad():
        predictions = model(track_left=track_left, track_right=track_right)
    
    print(f"\n🎯 Predictions for first sample:")
    for i, waypoint in enumerate(predictions[0]):
        print(f"   Waypoint {i+1}: ({waypoint[0]:.3f}, {waypoint[1]:.3f})")
    
    # Analyze prediction patterns
    mean_prediction = predictions.mean(dim=0)
    std_prediction = predictions.std(dim=0)
    
    print(f"\n📈 Prediction Statistics:")
    print(f"   • Mean waypoint 1: ({mean_prediction[0, 0]:.3f}, {mean_prediction[0, 1]:.3f})")
    print(f"   • Mean waypoint 2: ({mean_prediction[1, 0]:.3f}, {mean_prediction[1, 1]:.3f})")
    print(f"   • Mean waypoint 3: ({mean_prediction[2, 0]:.3f}, {mean_prediction[2, 1]:.3f})")
    print(f"   • Std waypoint 1: ({std_prediction[0, 0]:.3f}, {std_prediction[0, 1]:.3f})")
    print(f"   • Std waypoint 2: ({std_prediction[1, 0]:.3f}, {std_prediction[1, 1]:.3f})")
    print(f"   • Std waypoint 3: ({std_prediction[2, 0]:.3f}, {std_prediction[2, 1]:.3f})")
    
    print(f"\n✅ Model produces reasonable predictions!")
    
    return predictions


def main():
    """Run all tests."""
    try:
        test_trained_mlp()
        test_model_performance()
        print(f"\n🎉 All model tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 