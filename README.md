# NN From PyTorch

A deep learning project exploring the fundamentals of neural networks by building a PyTorch Multi-Layer Perceptron (MLP) from scratch to classify complex, non-linear spiral datasets. 

## Features
- **Modular Architecture:** Extracted from a Jupyter Notebook into production-ready Python scripts (`src/`).
- **Vanishing Gradient Fixes:** Implements a deep 25-layer network using modern techniques to maintain stable gradients:
  - ResNet-style Skip Connections
  - Batch Normalization (`BatchNorm1d`)
  - He (Kaiming) Weight Initialization
- **Custom Datasets:** Includes generators for complex, intertwined spiral patterns to test network capacity and decision boundaries.

## How to Run
Ensure you have PyTorch, scikit-learn, and matplotlib installed.

```bash
# Run the main training loop and evaluate the model
python src/train.py
```
