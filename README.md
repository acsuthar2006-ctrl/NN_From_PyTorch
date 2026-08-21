---
tags:
- image-classification
- pytorch
- resnet9
- custom-architecture
datasets:
- 
metrics:
- accuracy
model-index:
- name: ResNet-Apex
  results:
  - task:
      type: image-classification
    dataset:
      name: CIFAR-10
      type: cifar10
    metrics:
      - type: accuracy
        value: 91.30
---

# Model Card for ResNet-Apex

ResNet-Apex is a highly optimized, custom 9-layer Convolutional Neural Network (ResNet9 architecture) trained entirely from scratch. It achieves **91.30% test accuracy** on the CIFAR-10 dataset using aggressive regularization and a long-tail cosine annealing schedule.

## Model Details

### Model Description

This model was built to explore the absolute limits of small architectures without relying on transfer learning. By utilizing a widened capacity (512-channel bottleneck) and heavy data augmentations, ResNet-Apex acts as a highly capable lightweight image classifier.

- **Developed by:** Aarya Suthar
- **Model type:** Convolutional Neural Network (ResNet9)
- **Parameters:** 6.57 Million (6,575,370)
- **Language(s):** Python (PyTorch)
- **License:** MIT

### Model Sources

- **Repository:** https://github.com/acsuthar2006-ctrl/ResNet-Apex

## Uses

### Direct Use

The model is intended for direct image classification on the CIFAR-10 dataset, categorizing 32x32 images into one of 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

### Out-of-Scope Use

This model is trained specifically on low-resolution 32x32 images and will not perform well on high-resolution real-world photography without significant resizing and domain adaptation.

## Bias, Risks, and Limitations

As the model is trained exclusively on the CIFAR-10 dataset, it inherits any class imbalances or biases present in the original dataset. It should be used for educational and benchmarking purposes rather than critical real-world deployments.

## Training Details

### Training Data

The model was trained on the standard **CIFAR-10** training split consisting of 50,000 32x32 color images. 

### Training Procedure 

#### Preprocessing & Data Augmentation
To prevent memorization across the 150-epoch training cycle, aggressive data augmentations were applied:
- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip(p=0.5)`
- `RandomErasing(p=0.1, scale=(0.02, 0.25))`
- Standardization `Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])`

#### Training Hyperparameters
- **Epochs:** 150
- **Optimizer:** AdamW
- **Learning Rate:** 0.001 (Starting)
- **Weight Decay:** 1e-4
- **Scheduler:** `CosineAnnealingLR`
- **Precision:** Mixed Precision (PyTorch AMP)
- **Batch Size:** 512

## Evaluation

### Testing Data & Metrics

The model was evaluated on the CIFAR-10 test split (10,000 images).

### Results

- **Test Accuracy:** 91.30%
- **Train Accuracy:** 98.38%
- **Final Test Loss:** 0.5094

## How to Get Started with the Model

Use the code below to load and run the model.

```python
import torch
from model import MyNeuralNet

# Initialize architecture
model = MyNeuralNet(num_classes=10)

# Load weights
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model.load_state_dict(torch.load("cifar10_model_long_run.pth", map_location=device))
model.eval()

print("Model loaded successfully!")
```

## Adapting to Custom Datasets (e.g., Clothing Classification)

Because this architecture has 6.57M parameters, it has extremely high capacity. As a general rule in deep learning: if a model has enough capacity to easily fit (and even overfit) a complex dataset like CIFAR-10, it is highly capable of mastering other visual domains like clothing classification (e.g., Fashion-MNIST). 

To train this architecture on your own dataset, you only need to change two things:
1. **Data Loading (`src/data.py`):** Swap out the CIFAR-10 dataset for your target dataset (using `torchvision.datasets.ImageFolder` or similar). Be sure to update the `Normalize` mean and standard deviation to match your new data.
2. **Output Classes:** Update the `num_classes` parameter when initializing the model in your training script to match the number of categories in your new dataset (e.g., `model = MyNeuralNet(num_classes=15)`).
