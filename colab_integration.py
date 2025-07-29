#!/usr/bin/env python3
"""
Colab Integration Script
Helps transition between local development and Google Colab for hybrid workflow.
"""

import os
import json
from pathlib import Path


def create_colab_notebook():
    """Create a Colab-ready notebook for our project."""
    
    colab_notebook = '''{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "setup"
      },
      "source": [
        "# Autonomous Driving with Transformers - Colab Version\\n",
        "\\n",
        "This notebook runs our trained models in Google Colab with GPU acceleration.\\n",
        "\\n",
        "## Setup Instructions\\n",
        "1. Enable GPU: Runtime > Change runtime type > T4 GPU\\n",
        "2. Run all cells in order\\n",
        "3. Download results when complete"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "git_setup"
      },
      "source": [
        "## Git Setup\\n",
        "Clone our repository from GitHub"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "clone_repo"
      },
      "outputs": [],
      "source": [
        "import os\\n",
        "\\n",
        "# Configure your GitHub details\\n",
        "os.environ['USER'] = 'nitin2468git'  # Your GitHub username\\n",
        "os.environ['REPO'] = 'autonomous-driving-transformers'  # Your repo name\\n",
        "os.environ['TOKEN'] = 'YOUR_GITHUB_TOKEN'  # Replace with your token\\n",
        "\\n",
        "# Clone the repository\\n",
        "%cd /content\\n",
        "!git clone https://${TOKEN}@github.com/${USER}/${REPO}.git\\n",
        "\\n",
        "# Navigate to project directory\\n",
        "%cd /content/autonomous-driving-transformers\\n",
        "!ls -la"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dataset_setup"
      },
      "source": [
        "## Dataset Setup\\n",
        "Download the SuperTuxKart dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "download_data"
      },
      "outputs": [],
      "source": [
        "# Download and extract dataset\\n",
        "!rm -rf drive_data\\n",
        "!curl -s -L https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip -o ./drive_data.zip && unzip -qo drive_data.zip\\n",
        "!ls -la drive_data/"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "install_deps"
      },
      "source": [
        "## Install Dependencies\\n",
        "Install PyTorch and other required packages"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "install_packages"
      },
      "outputs": [],
      "source": [
        "# Install PyTorch for GPU\\n",
        "!pip install torch torchvision torchaudio\\n",
        "\\n",
        "# Install other dependencies\\n",
        "!pip install -r requirements.txt\\n",
        "\\n",
        "# Verify GPU access\\n",
        "import torch\\n",
        "print(f'PyTorch version: {torch.__version__}')\\n",
        "print(f'CUDA available: {torch.cuda.is_available()}')\\n",
        "if torch.cuda.is_available():\\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "test_models"
      },
      "source": [
        "## Test Our Models\\n",
        "Verify our trained models work in Colab"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "test_local_models"
      },
      "outputs": [],
      "source": [
        "# Test our models\\n",
        "!python3 test_models.py\\n",
        "!python3 test_trained_model.py"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "train_models"
      },
      "source": [
        "## Train Models (Optional)\\n",
        "If you want to retrain models with GPU acceleration"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "train_with_gpu"
      },
      "outputs": [],
      "source": [
        "# Train MLP Planner with GPU\\n",
        "!python3 train_with_visualizations.py --model mlp_planner --epochs 50 --batch_size 32\\n",
        "\\n",
        "# Train Transformer Planner with GPU\\n",
        "!python3 train_with_visualizations.py --model transformer_planner --epochs 100 --batch_size 16\\n",
        "\\n",
        "# Train CNN Planner with GPU\\n",
        "!python3 train_with_visualizations.py --model cnn_planner --epochs 80 --batch_size 16"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "grade_models"
      },
      "source": [
        "## Grade Models\\n",
        "Run the grader to evaluate our models"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "run_grader"
      },
      "outputs": [],
      "source": [
        "# Grade our models\\n",
        "!python3 -m grader homework -vv --disable_color"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "supertux_setup"
      },
      "source": [
        "## SuperTuxKart Setup (Optional)\\n",
        "Install SuperTuxKart for driving visualization"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "install_supertux"
      },
      "outputs": [],
      "source": [
        "# Install SuperTuxKart\\n",
        "!sudo DEBIAN_FRONTEND=noninteractive apt install -qq libnvidia-gl-535\\n",
        "!pip install PySuperTuxKartData --index-url=https://www.cs.utexas.edu/~bzhou/dl_class/pystk\\n",
        "!pip install PySuperTuxKart --index-url=https://www.cs.utexas.edu/~bzhou/dl_class/pystk"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "driving_test"
      },
      "source": [
        "## Driving Test\\n",
        "Watch our models drive in SuperTuxKart!"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "run_driving"
      },
      "outputs": [],
      "source": [
        "# Test MLP Planner driving\\n",
        "!python3 -m homework.supertux_utils.evaluate --model mlp_planner --track lighthouse --max-steps 100\\n",
        "\\n",
        "# Test Transformer Planner driving\\n",
        "!python3 -m homework.supertux_utils.evaluate --model transformer_planner --track lighthouse --max-steps 100\\n",
        "\\n",
        "# Test CNN Planner driving\\n",
        "!python3 -m homework.supertux_utils.evaluate --model cnn_planner --track lighthouse --max-steps 100"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "visualize_results"
      },
      "source": [
        "## Visualize Results\\n",
        "Display driving videos and training plots"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "show_videos"
      },
      "outputs": [],
      "source": [
        "from IPython.display import Video, Image\\n",
        "from pathlib import Path\\n",
        "import matplotlib.pyplot as plt\\n",
        "import matplotlib.image as mpimg\\n",
        "\\n",
        "# Display training visualizations\\n",
        "outputs_dir = Path('outputs')\\n",
        "\\n",
        "for model_dir in ['mlp_planner', 'transformer_planner', 'cnn_planner']:\\n",
        "    model_path = outputs_dir / model_dir\\n",
        "    if model_path.exists():\\n",
        "        print(f'\\\\n📊 {model_dir.upper()} VISUALIZATIONS:')\\n",
        "        \\n",
        "        # Show training curves\\n",
        "        training_curves = list(model_path.glob('training_curves/*.png'))\\n",
        "        if training_curves:\\n",
        "            img = mpimg.imread(training_curves[0])\\n",
        "            plt.figure(figsize=(10, 6))\\n",
        "            plt.imshow(img)\\n",
        "            plt.axis('off')\\n",
        "            plt.title(f'{model_dir} Training Curves')\\n",
        "            plt.show()\\n",
        "        \\n",
        "        # Show error analysis\\n",
        "        error_plots = list(model_path.glob('error_analysis/*.png'))\\n",
        "        if error_plots:\\n",
        "            img = mpimg.imread(error_plots[0])\\n",
        "            plt.figure(figsize=(10, 6))\\n",
        "            plt.imshow(img)\\n",
        "            plt.axis('off')\\n",
        "            plt.title(f'{model_dir} Error Analysis')\\n",
        "            plt.show()\\n",
        "\\n",
        "# Display driving videos\\n",
        "video_dir = Path('videos')\\n",
        "if video_dir.exists():\\n",
        "    print('\\\\n🎮 DRIVING VIDEOS:')\\n",
        "    for video_path in sorted(video_dir.glob('*.mp4')):\\n",
        "        print(f'📹 {video_path.name}')\\n",
        "        display(Video(str(video_path), embed=True))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "commit_changes"
      },
      "source": [
        "## Commit Changes\\n",
        "Push any changes back to GitHub"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "git_commit"
      },
      "outputs": [],
      "source": [
        "# Commit and push changes\\n",
        "!git status\\n",
        "!git add homework/*.py outputs/\\n",
        "!git config --global user.email 'your_email@example.com'\\n",
        "!git config --global user.name 'Your Name'\\n",
        "!git commit -m 'Colab training results'\\n",
        "!git push origin main"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "download_results"
      },
      "source": [
        "## Download Results\\n",
        "Download trained models and visualizations"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "download_files"
      },
      "outputs": [],
      "source": [
        "# Create zip file for download\\n",
        "!zip -r colab_results.zip homework/*.th outputs/ videos/\\n",
        "print('\\\\n📦 Download colab_results.zip from the file browser on the left!')"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "Autonomous Driving with Transformers",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}'''
    
    # Save the notebook
    with open('autonomous_driving_colab.ipynb', 'w') as f:
        f.write(colab_notebook)
    
    print("✅ Created autonomous_driving_colab.ipynb")
    print("📝 Instructions:")
    print("   1. Upload this notebook to Google Colab")
    print("   2. Enable T4 GPU runtime")
    print("   3. Replace 'YOUR_GITHUB_TOKEN' with your actual token")
    print("   4. Run all cells in order")


def create_colab_setup_script():
    """Create a setup script for Colab environment."""
    
    setup_script = '''#!/bin/bash
# Colab Setup Script
# Run this in Colab to set up the environment

echo "🚀 Setting up Autonomous Driving Environment in Colab"

# Install system dependencies
sudo apt-get update
sudo apt-get install -y git curl unzip

# Install PyTorch for GPU
pip install torch torchvision torchaudio

# Install other dependencies
pip install matplotlib numpy tqdm tensorboard

# Clone repository (replace with your details)
git clone https://github.com/nitin2468git/autonomous-driving-transformers.git
cd autonomous-driving-transformers

# Download dataset
curl -s -L https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip -o ./drive_data.zip
unzip -qo drive_data.zip

# Test setup
python3 test_models.py

echo "✅ Setup complete! Ready to train models."
'''
    
    with open('colab_setup.sh', 'w') as f:
        f.write(setup_script)
    
    print("✅ Created colab_setup.sh")
    print("📝 Run this in Colab terminal to set up environment")


def main():
    """Create Colab integration files."""
    print("🎯 Creating Colab Integration Files")
    print("=" * 50)
    
    create_colab_notebook()
    create_colab_setup_script()
    
    print("\n📋 Hybrid Workflow Instructions:")
    print("=" * 50)
    print("1. 🏠 LOCAL DEVELOPMENT:")
    print("   • Develop and test models locally")
    print("   • Use git for version control")
    print("   • Debug and iterate quickly")
    
    print("\n2. ☁️ COLAB ACCELERATION:")
    print("   • Upload autonomous_driving_colab.ipynb to Colab")
    print("   • Enable T4 GPU runtime")
    print("   • Run intensive training with GPU")
    print("   • Download results back to local")
    
    print("\n3. 🔄 SYNC WORKFLOW:")
    print("   • git push from local")
    print("   • git pull in Colab")
    print("   • Train with GPU")
    print("   • Download results")
    print("   • git pull back to local")
    
    print("\n🎯 Benefits:")
    print("   ✅ Local: Fast iteration, git integration, debugging")
    print("   ✅ Colab: GPU acceleration, SuperTuxKart, visualization")
    print("   ✅ Hybrid: Best of both worlds!")


if __name__ == "__main__":
    main() 