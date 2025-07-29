#!/usr/bin/env python3
"""
Visualization script for training analysis.
Generates plots and saves them to organized output folders.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import torch


def plot_training_curves(train_losses, val_losses, model_name, save_path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title(f'{model_name} - Training Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved training curves: {save_path}")


def plot_error_analysis(longitudinal_errors, lateral_errors, model_name, save_path):
    """Plot longitudinal and lateral error analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(longitudinal_errors) + 1)
    
    # Longitudinal Error
    ax1.plot(epochs, longitudinal_errors, 'g-', linewidth=2)
    ax1.set_title('Longitudinal Error', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error')
    ax1.grid(True, alpha=0.3)
    
    # Lateral Error
    ax2.plot(epochs, lateral_errors, 'm-', linewidth=2)
    ax2.set_title('Lateral Error', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved error analysis: {save_path}")


def plot_model_comparison(model_results, save_path):
    """Compare final performance across models."""
    models = list(model_results.keys())
    longitudinal_errors = [model_results[m]['longitudinal_error'] for m in models]
    lateral_errors = [model_results[m]['lateral_error'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Longitudinal Error Comparison
    bars1 = ax1.bar(x - width/2, longitudinal_errors, width, label='Longitudinal', color='skyblue')
    ax1.set_title('Longitudinal Error Comparison', fontweight='bold')
    ax1.set_ylabel('Error')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Lateral Error Comparison
    bars2 = ax2.bar(x + width/2, lateral_errors, width, label='Lateral', color='lightcoral')
    ax2.set_title('Lateral Error Comparison', fontweight='bold')
    ax2.set_ylabel('Error')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.legend()
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved model comparison: {save_path}")


def create_sample_visualizations():
    """Create sample visualizations for demonstration."""
    print("🎨 Creating sample visualizations...")
    
    # Sample data (replace with real training data later)
    epochs = 20
    train_losses = [2.5 - 0.1*i + np.random.normal(0, 0.1) for i in range(epochs)]
    val_losses = [2.3 - 0.08*i + np.random.normal(0, 0.15) for i in range(epochs)]
    
    longitudinal_errors = [0.4 - 0.02*i + np.random.normal(0, 0.05) for i in range(epochs)]
    lateral_errors = [1.5 - 0.05*i + np.random.normal(0, 0.1) for i in range(epochs)]
    
    # Create outputs directory structure
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # MLP Planner visualizations
    mlp_dir = outputs_dir / "mlp_planner"
    mlp_dir.mkdir(exist_ok=True)
    (mlp_dir / "training_curves").mkdir(exist_ok=True)
    (mlp_dir / "error_analysis").mkdir(exist_ok=True)
    
    plot_training_curves(
        train_losses, val_losses, 
        "MLP Planner",
        mlp_dir / "training_curves" / "mlp_training_curves.png"
    )
    
    plot_error_analysis(
        longitudinal_errors, lateral_errors,
        "MLP Planner", 
        mlp_dir / "error_analysis" / "mlp_error_analysis.png"
    )
    
    # Model comparison
    (outputs_dir / "model_comparison").mkdir(exist_ok=True)
    
    model_results = {
        "MLP Planner": {"longitudinal_error": 0.15, "lateral_error": 0.45},
        "Transformer Planner": {"longitudinal_error": 0.12, "lateral_error": 0.38},
        "CNN Planner": {"longitudinal_error": 0.25, "lateral_error": 0.42}
    }
    
    plot_model_comparison(
        model_results,
        outputs_dir / "model_comparison" / "model_performance_comparison.png"
    )
    
    print("✅ Sample visualizations created!")


if __name__ == "__main__":
    create_sample_visualizations() 