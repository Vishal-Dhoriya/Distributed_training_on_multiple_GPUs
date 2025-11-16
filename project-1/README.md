
# Multi-GPU Mirrored Strategy

Distributed training implementation using TensorFlow's MirroredStrategy for Fashion MNIST classification.

## Overview

This project demonstrates distributed training across multiple GPUs (or single GPU/CPU fallback) using TensorFlow's MirroredStrategy with custom training loops.

## Project Structure

- `utils.py` - Data loading, preprocessing, and model creation
- `trainer.py` - Training and test step functions with distributed logic
- `main.py` - Orchestration and entry point

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy

## Installation

```bash
pip install tensorflow numpy
```

## Usage

Run the training pipeline:

```bash
python main.py
```

## Features

- Automatic GPU detection (multi-GPU, single GPU, or CPU fallback)
- Custom distributed training loops
- Separate data preprocessing and model training modules
- Clean, modular code structure

## Model Architecture

- 2x Conv2D layers (32, 64 filters)
- 2x MaxPooling layers
- Dense layers (64 units, 10 output classes)

## Dataset

Fashion MNIST - 60,000 training and 10,000 test images (28x28 grayscale)

## Training Configuration

- Epochs: 10
- Batch size: 64 per replica
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

## Notes

- Works with single or multiple GPUs
- Falls back to CPU if GPU unavailable
<!-- - TensorFlow logs suppressed for cleaner output -->
