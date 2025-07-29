"""
Usage:
    python3 -m homework.train_planner --model mlp_planner --epochs 50 --batch_size 32
    python3 -m homework.train_planner --model transformer_planner --epochs 100 --batch_size 16
    python3 -m homework.train_planner --model cnn_planner --epochs 80 --batch_size 16
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework.models import MODEL_FACTORY, save_model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    metric: PlannerMetric,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metric.reset()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        if 'image' in batch:
            # CNN model
            images = batch['image'].to(device)
            waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            # Forward pass
            outputs = model(images)
        else:
            # MLP/Transformer models
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            # Forward pass
            outputs = model(track_left=track_left, track_right=track_right)
        
        # Compute loss
        loss = criterion(outputs, waypoints)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        metric.add(outputs, waypoints, waypoints_mask)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Compute final metrics
    metrics = metric.compute()
    avg_loss = total_loss / len(train_loader)
    
    print(f"Train - Loss: {avg_loss:.4f}, "
          f"Longitudinal Error: {metrics['longitudinal_error']:.4f}, "
          f"Lateral Error: {metrics['lateral_error']:.4f}")
    
    return avg_loss


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric: PlannerMetric,
) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    metric.reset()
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            if 'image' in batch:
                # CNN model
                images = batch['image'].to(device)
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                
                # Forward pass
                outputs = model(images)
            else:
                # MLP/Transformer models
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                
                # Forward pass
                outputs = model(track_left=track_left, track_right=track_right)
            
            # Compute loss
            loss = criterion(outputs, waypoints)
            
            # Update metrics
            total_loss += loss.item()
            metric.add(outputs, waypoints, waypoints_mask)
    
    # Compute final metrics
    metrics = metric.compute()
    avg_loss = total_loss / len(val_loader)
    
    print(f"Val - Loss: {avg_loss:.4f}, "
          f"Longitudinal Error: {metrics['longitudinal_error']:.4f}, "
          f"Lateral Error: {metrics['lateral_error']:.4f}")
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train planner models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['mlp_planner', 'transformer_planner', 'cnn_planner'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data workers')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training {args.model} for {args.epochs} epochs")
    
    # Create model
    model_class = MODEL_FACTORY[args.model]
    model = model_class().to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Setup data
    if args.model == 'cnn_planner':
        # CNN uses images, so use default transform
        train_transform = "default"
        val_transform = "default"
    else:
        # MLP/Transformer use only track data
        train_transform = "state_only"
        val_transform = "state_only"
    
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=train_transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=val_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()
    
    # Training loop
    best_val_loss = float('inf')
    best_val_metrics = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, train_metric)
        train_time = time.time() - start_time
        
        # Validate
        start_time = time.time()
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, val_metric)
        val_time = time.time() - start_time
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            save_model(model)
            print(f"Saved new best model with val loss: {val_loss:.4f}")
        
        print(f"Train time: {train_time:.2f}s, Val time: {val_time:.2f}s")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best validation metrics:")
    print(f"  Longitudinal Error: {best_val_metrics['longitudinal_error']:.4f}")
    print(f"  Lateral Error: {best_val_metrics['lateral_error']:.4f}")
    print(f"  Combined L1 Error: {best_val_metrics['l1_error']:.4f}")
    
    # Check if model meets requirements
    if args.model in ['mlp_planner', 'transformer_planner']:
        if (best_val_metrics['longitudinal_error'] < 0.2 and 
            best_val_metrics['lateral_error'] < 0.6):
            print("✅ Model meets performance requirements!")
        else:
            print("❌ Model does not meet performance requirements.")
    elif args.model == 'cnn_planner':
        if (best_val_metrics['longitudinal_error'] < 0.30 and 
            best_val_metrics['lateral_error'] < 0.45):
            print("✅ Model meets performance requirements!")
        else:
            print("❌ Model does not meet performance requirements.")


if __name__ == "__main__":
    main()
