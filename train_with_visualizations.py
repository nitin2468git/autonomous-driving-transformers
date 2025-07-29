#!/usr/bin/env python3
"""
Enhanced training script with detailed visualizations and explanations.
This will help us understand what's happening during training.
"""

import argparse
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework.models import MODEL_FACTORY, save_model
from visualize_training import plot_training_curves, plot_error_analysis


def explain_training_setup(model_name, epochs, batch_size, lr):
    """Explain the training setup in simple terms."""
    print("\n" + "="*60)
    print("🚗 TRAINING SETUP EXPLANATION")
    print("="*60)
    
    print(f"📊 Model: {model_name}")
    print(f"⏱️  Training for {epochs} epochs")
    print(f"📦 Batch size: {batch_size} (samples processed together)")
    print(f"📈 Learning rate: {lr} (how big steps to take)")
    
    print("\n🎯 What we're trying to achieve:")
    if model_name in ['mlp_planner', 'transformer_planner']:
        print("   • Longitudinal Error < 0.2 (speed prediction)")
        print("   • Lateral Error < 0.6 (steering prediction)")
    else:
        print("   • Longitudinal Error < 0.30 (speed prediction)")
        print("   • Lateral Error < 0.45 (steering prediction)")
    
    print("\n📈 What we'll monitor:")
    print("   • Training Loss: How much error during training")
    print("   • Validation Loss: How well it generalizes")
    print("   • Longitudinal Error: Forward/backward accuracy")
    print("   • Lateral Error: Left/right accuracy")
    print("="*60)


def explain_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics):
    """Explain what happened in this epoch."""
    print(f"\n📊 EPOCH {epoch} RESULTS:")
    print("-" * 40)
    
    print(f"🎯 Training Loss: {train_loss:.4f}")
    print(f"✅ Validation Loss: {val_loss:.4f}")
    
    print(f"\n📈 Training Metrics:")
    print(f"   • Longitudinal Error: {train_metrics['longitudinal_error']:.4f}")
    print(f"   • Lateral Error: {train_metrics['lateral_error']:.4f}")
    
    print(f"\n📈 Validation Metrics:")
    print(f"   • Longitudinal Error: {val_metrics['longitudinal_error']:.4f}")
    print(f"   • Lateral Error: {val_metrics['lateral_error']:.4f}")
    
    # Check if we're meeting requirements
    if val_metrics['longitudinal_error'] < 0.2 and val_metrics['lateral_error'] < 0.6:
        print("🎉 EXCELLENT! Model meets performance requirements!")
    elif val_metrics['longitudinal_error'] < 0.3 and val_metrics['lateral_error'] < 0.8:
        print("👍 GOOD! Model is improving and getting close!")
    else:
        print("📈 KEEP TRAINING! Model needs more epochs to improve.")


def train_with_visualizations(
    model_name: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_plots: bool = True
):
    """Train model with detailed visualizations and explanations."""
    
    # Explain what we're about to do
    explain_training_setup(model_name, epochs, batch_size, lr)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n💻 Using device: {device}")
    
    # Create model
    model_class = MODEL_FACTORY[model_name]
    model = model_class().to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")
    
    # Setup data
    if model_name == 'cnn_planner':
        transform = "default"  # Include images
    else:
        transform = "state_only"  # Only road boundaries
    
    print(f"\n📂 Loading data with transform: {transform}")
    
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # For debugging
    )
    
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Metrics tracking
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()
    
    # History for plotting
    train_losses = []
    val_losses = []
    train_longitudinal_errors = []
    train_lateral_errors = []
    val_longitudinal_errors = []
    val_lateral_errors = []
    
    # Training loop
    best_val_loss = float('inf')
    best_val_metrics = None
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print("="*60)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_metric.reset()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            if 'image' in batch:
                images = batch['image'].to(device)
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                outputs = model(images)
            else:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                outputs = model(track_left=track_left, track_right=track_right)
            
            # Compute loss
            loss = criterion(outputs, waypoints)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_metric.add(outputs, waypoints, waypoints_mask)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metric.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                if 'image' in batch:
                    images = batch['image'].to(device)
                    waypoints = batch['waypoints'].to(device)
                    waypoints_mask = batch['waypoints_mask'].to(device)
                    outputs = model(images)
                else:
                    track_left = batch['track_left'].to(device)
                    track_right = batch['track_right'].to(device)
                    waypoints = batch['waypoints'].to(device)
                    waypoints_mask = batch['waypoints_mask'].to(device)
                    outputs = model(track_left=track_left, track_right=track_right)
                
                loss = criterion(outputs, waypoints)
                val_loss += loss.item()
                val_metric.add(outputs, waypoints, waypoints_mask)
        
        # Compute metrics
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_longitudinal_errors.append(train_metrics['longitudinal_error'])
        train_lateral_errors.append(train_metrics['lateral_error'])
        val_longitudinal_errors.append(val_metrics['longitudinal_error'])
        val_lateral_errors.append(val_metrics['lateral_error'])
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = val_metrics
            save_model(model)
            print(f"💾 Saved new best model!")
        
        # Explain results
        explain_epoch_results(epoch + 1, avg_train_loss, avg_val_loss, train_metrics, val_metrics)
        
        # Generate plots every 10 epochs or at the end
        if save_plots and (epoch % 10 == 9 or epoch == epochs - 1):
            print(f"\n📊 Generating visualizations...")
            
            # Create output directory
            output_dir = Path("outputs") / model_name
            output_dir.mkdir(exist_ok=True)
            (output_dir / "training_curves").mkdir(exist_ok=True)
            (output_dir / "error_analysis").mkdir(exist_ok=True)
            
            # Plot training curves
            plot_training_curves(
                train_losses, val_losses,
                model_name.replace('_', ' ').title(),
                output_dir / "training_curves" / f"{model_name}_training_curves.png"
            )
            
            # Plot error analysis
            plot_error_analysis(
                val_longitudinal_errors, val_lateral_errors,
                model_name.replace('_', ' ').title(),
                output_dir / "error_analysis" / f"{model_name}_error_analysis.png"
            )
        
        epoch_time = time.time() - start_time
        print(f"⏱️  Epoch time: {epoch_time:.2f}s")
        print(f"📈 Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)
    
    # Final results
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation metrics:")
    print(f"  • Longitudinal Error: {best_val_metrics['longitudinal_error']:.4f}")
    print(f"  • Lateral Error: {best_val_metrics['lateral_error']:.4f}")
    print(f"  • Combined L1 Error: {best_val_metrics['l1_error']:.4f}")
    
    # Check requirements
    if model_name in ['mlp_planner', 'transformer_planner']:
        if (best_val_metrics['longitudinal_error'] < 0.2 and 
            best_val_metrics['lateral_error'] < 0.6):
            print("🎉 SUCCESS! Model meets all requirements!")
        else:
            print("📈 Model needs more training or hyperparameter tuning.")
    else:
        if (best_val_metrics['longitudinal_error'] < 0.30 and 
            best_val_metrics['lateral_error'] < 0.45):
            print("🎉 SUCCESS! Model meets all requirements!")
        else:
            print("📈 Model needs more training or hyperparameter tuning.")
    
    return best_val_metrics


def main():
    parser = argparse.ArgumentParser(description='Train with detailed visualizations')
    parser.add_argument('--model', type=str, default='mlp_planner',
                       choices=['mlp_planner', 'transformer_planner', 'cnn_planner'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no_plots', action='store_true', help='Disable plot generation')
    
    args = parser.parse_args()
    
    train_with_visualizations(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_plots=not args.no_plots
    )


if __name__ == "__main__":
    main() 