# Distributed Training with Transfer Learning

Distributed training implementation using TensorFlow's MirroredStrategy on Oxford Flowers 102 dataset with ResNet50 transfer learning.

## Overview

This project demonstrates distributed training across multiple GPUs (or single GPU/CPU fallback) using TensorFlow's MirroredStrategy with custom training loops and transfer learning.

## Project Structure

- `config.py` - Configuration, constants, and dataset loading
- `model_utils.py` - Model creation, utilities, and preprocessing
- `trainer.py` - Training and test step functions with distributed logic
- `main.py` - Orchestration and entry point

## Requirements

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Datasets
- TensorFlow Hub
- NumPy
- tqdm

## Installation

```bash
pip install tensorflow tensorflow-datasets tensorflow-hub numpy tqdm
```

## Usage

Run the training pipeline:

```bash
python main.py
```

## Features

- Automatic GPU detection (multi-GPU, single GPU, or CPU fallback)
- Custom distributed training loops
- Transfer learning with ResNet50 from TensorFlow Hub
- Separate data preprocessing and model training modules
- Clean, modular code structure
- Progress tracking with tqdm

## Model Architecture

- ResNet50 feature extractor (pretrained, frozen)
- Dense classifier layer (softmax activation)
- Classes: 102 flower categories

## Dataset

Oxford Flowers 102 - Flower classification dataset
- Training: 80% of data
- Validation: 10% of data
- Test: 10% of data
- Image size: 224x224 pixels
- Classes: 102

## Training Configuration

- Epochs: 10
- Batch size: 64 per replica
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Learning rate: Default Adam (0.001)

## Notes

- Works with single or multiple GPUs
- Falls back to CPU if GPU unavailable
- ResNet50 model automatically downloaded from TensorFlow Hub on first run
- Dataset automatically downloaded from TensorFlow Datasets on first run
<!-- - TensorFlow logs suppressed for cleaner output -->

