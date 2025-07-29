# Autonomous Driving with Transformers and CNNs

This repository contains implementations of three different neural network architectures for autonomous driving trajectory prediction using the SuperTuxKart Drive Dataset.

## 🚗 Project Overview

The goal is to predict future waypoints (trajectory) for autonomous driving using three different approaches:

1. **MLP Planner**: Simple feedforward neural network using ground truth lane boundaries
2. **Transformer Planner**: Cross-attention based model inspired by Perceiver architecture
3. **CNN Planner**: Convolutional neural network that predicts waypoints directly from images

## 📊 Performance Requirements

| Model | Longitudinal Error | Lateral Error |
|-------|-------------------|---------------|
| MLP Planner | < 0.2 | < 0.6 |
| Transformer Planner | < 0.2 | < 0.6 |
| CNN Planner | < 0.30 | < 0.45 |

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4) for optimal performance
- Miniconda or Anaconda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nitin2468git/autonomous-driving-transformers.git
cd autonomous-driving-transformers
```

2. **Create and activate conda environment**
```bash
conda create -n hw4 python=3.10 -y
conda activate hw4
```

3. **Install PyTorch for Apple Silicon**
```bash
pip install torch torchvision torchaudio
```

4. **Install other dependencies**
```bash
pip install -r requirements.txt
```

5. **Download the dataset**
```bash
curl -s -L https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip -o ./drive_data.zip && unzip -qo drive_data.zip
```

## 🏃‍♂️ Training

### MLP Planner
```bash
python3 -m homework.train_planner --model mlp_planner --epochs 50 --batch_size 32
```

### Transformer Planner
```bash
python3 -m homework.train_planner --model transformer_planner --epochs 100 --batch_size 16
```

### CNN Planner
```bash
python3 -m homework.train_planner --model cnn_planner --epochs 80 --batch_size 16
```

### Training Options
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--num_workers`: Data loading workers (default: 2)
- `--device`: Device to use (default: auto-detects MPS/CPU)

## 🧠 Model Architectures

### MLP Planner
- **Input**: Ground truth lane boundaries (`track_left`, `track_right`)
- **Architecture**: 4-layer feedforward network with ReLU activations and dropout
- **Output**: Predicted waypoints `(batch_size, n_waypoints, 2)`

### Transformer Planner
- **Input**: Ground truth lane boundaries (`track_left`, `track_right`)
- **Architecture**: Cross-attention with learned query embeddings (Perceiver-inspired)
- **Components**: 
  - Learned waypoint query embeddings
  - Linear projection of track points
  - Transformer decoder layer for cross-attention
- **Output**: Predicted waypoints `(batch_size, n_waypoints, 2)`

### CNN Planner
- **Input**: RGB images `(batch_size, 3, 96, 128)`
- **Architecture**: 4-layer CNN with batch normalization and pooling
- **Components**:
  - Convolutional backbone (32→64→128→256 channels)
  - Max pooling layers
  - Fully connected layers for waypoint prediction
- **Output**: Predicted waypoints `(batch_size, n_waypoints, 2)`

## 📈 Evaluation

### Offline Metrics
- **Longitudinal Error**: Absolute difference in forward direction (speed prediction)
- **Lateral Error**: Absolute difference in left/right direction (steering prediction)

### Grading
```bash
python3 -m grader homework -v
```

## 🎮 SuperTuxKart Integration (Optional)

For visualization of driving performance:

```bash
pip install PySuperTuxKartData
pip install PySuperTuxKart --index-url=https://www.cs.utexas.edu/~bzhou/dl_class/pystk

# Run evaluation
python3 -m homework.supertux_utils.evaluate --model mlp_planner --track lighthouse
```

## 📁 Project Structure

```
autonomous-driving-transformers/
├── homework/
│   ├── models.py              # Model implementations
│   ├── train_planner.py       # Training pipeline
│   ├── metrics.py             # Evaluation metrics
│   └── datasets/              # Data loading and transforms
├── grader/                    # Grading system
├── drive_data/                # SuperTuxKart dataset
├── assets/                    # Project images
└── requirements.txt           # Dependencies
```

## 🔧 Key Features

- **Apple Silicon Optimized**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **Modular Design**: Easy to extend with new model architectures
- **Comprehensive Training**: Includes validation, learning rate scheduling, and model saving
- **Performance Tracking**: Real-time metrics during training
- **Git Integration**: Ready for version control and collaboration

## 📝 Usage Examples

### Quick Test
```bash
# Test MLP model with 2 epochs
python3 -m homework.train_planner --model mlp_planner --epochs 2 --batch_size 8 --num_workers 0
```

### Full Training
```bash
# Train Transformer model with custom parameters
python3 -m homework.train_planner \
    --model transformer_planner \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4 \
    --weight_decay 1e-5
```

## 🤝 Contributing

This is a university assignment, but feel free to:
- Report issues
- Suggest improvements
- Fork for your own projects

## 📄 License

This project is part of the UT Austin Deep Learning course curriculum.

## 🙏 Acknowledgments

- UT Austin Deep Learning Course
- SuperTuxKart for the driving simulation
- PyTorch team for the excellent framework

---

**Note**: This project is designed for educational purposes and demonstrates various approaches to autonomous driving trajectory prediction using modern deep learning techniques.
