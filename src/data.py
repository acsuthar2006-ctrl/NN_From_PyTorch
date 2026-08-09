import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def make_spirals(n_samples=300, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
    
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    data_a += np.random.randn(n, 2) * noise
    
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    data_b += np.random.randn(n, 2) * noise
    
    X = np.vstack([data_a, data_b])
    y = np.zeros(n_samples, dtype=int)
    y[n:] = 1
    
    return X, y

def make_complex_spirals(n_samples=3000, noise=0.5, rotations=3, random_state=None):
    """
    Generates a much more complex dataset where the spirals wrap around each other
    multiple times (controlled by 'rotations'), creating a highly non-linear
    decision boundary that requires a deeper/smarter network to solve.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n = n_samples // 2
    
    # Increase the maximum angle to make the spirals wrap around more
    theta = np.sqrt(np.random.rand(n)) * rotations * 2 * np.pi
    
    # Class 0
    r_a = theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    data_a += np.random.randn(n, 2) * noise
    
    # Class 1
    r_b = -theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    data_b += np.random.randn(n, 2) * noise
    
    X = np.vstack([data_a, data_b])
    y = np.zeros(n_samples, dtype=int)
    y[n:] = 1
    
    return X, y

def get_dataloaders(n_samples=3000, noise=0.8, random_state=42, batch_size=256, use_complex=True):
    # Generate raw data (using complex spirals by default now!)
    if use_complex:
        X, y = make_complex_spirals(n_samples=n_samples, noise=noise, rotations=3, random_state=random_state)
    else:
        X, y = make_spirals(n_samples=n_samples, noise=noise, random_state=random_state)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, X, y
