#!/bin/bash
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
