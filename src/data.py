import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(batch_size=128):
    """
    Downloads and loads the MNIST handwritten digits dataset.
    Returns train_loader and test_loader.
    """
    # Define transformations for the images:
    # 1. Convert to PyTorch Tensor
    # 2. Normalize with the known mean and standard deviation of MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )

    # Download and load the testing data
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Create iterators for the data
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    return train_loader, test_loader
