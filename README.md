# NN From PyTorch

A deep learning project exploring the fundamentals of neural networks by building a PyTorch Multi-Layer Perceptron (MLP) from scratch to classify the MNIST handwritten digit dataset. 

## Features
- **Deep ResNet-style MLP:** Achieves ~98% test accuracy on MNIST without using Convolutional Neural Networks (CNNs).
- **Vanishing Gradient Fixes:** Implements a deep 10-layer network using modern techniques to maintain stable gradients:
  - ResNet-style Skip Connections
  - Batch Normalization (`BatchNorm1d`)
  - He (Kaiming) Weight Initialization
- **Data Pipeline:** Integrates `torchvision.datasets` for efficient downloading, normalization, and batching of the MNIST dataset.
- **Modular Architecture:** Cleanly separated into data loading, model definition, training loop, and evaluation utilities (`src/`).

## How to Run
Ensure you have PyTorch, torchvision, and matplotlib installed.

```bash
# Run the main training loop and evaluate the model
python src/train.py
```
