# ResNet-Apex

A custom-built, highly optimized 9-layer Convolutional Neural Network (ResNet9 architecture) trained entirely from scratch to achieve **91.3% test accuracy** on the CIFAR-10 dataset.

## Features
- **Custom ResNet9 Architecture:** Optimized 512-channel width with residual blocks, adaptive pooling, and He (Kaiming) weight initialization.
- **Advanced Regularization:** Employs Dropout (0.2) and Weight Decay (1e-4) to combat overfitting without crippling model capacity.
- **Heavy Data Augmentation:** Prevents image memorization across long training runs using `RandomHorizontalFlip`, `RandomCrop(32, padding=4)`, and `RandomErasing`.
- **Cosine Annealing Scheduler:** Uses a 150-epoch long training schedule with `CosineAnnealingLR` and `AdamW` to slowly decay the learning rate and squeeze out maximum test accuracy.
- **Cloud/Kaggle Ready:** Fully supports multi-GPU training (`nn.DataParallel`) and PyTorch AMP (Automatic Mixed Precision) for blazing-fast execution.

## The Journey to 91.3%
Getting past 90% on CIFAR-10 with a tiny network trained from scratch is a significant milestone. This repo demonstrates the deep-learning trade-offs between hyper-fast super convergence (using OneCycleLR for 40 epochs to get ~90.7%) vs. slow, stable convergence (using CosineAnnealingLR for 150 epochs to break **91.3%**).

## How to Run
Ensure you have PyTorch, torchvision, and matplotlib installed.

```bash
# Run the main training loop and evaluate the model
python src/train.py

# Generate and visualize predictions from the saved model
python src/predict.py
```
