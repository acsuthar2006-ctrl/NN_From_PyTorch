import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total

def plot_decision_boundary(model, X, y, device):
    model.eval()
    
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    grid = grid.to(device)
    
    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits)
    
    Z = probs.cpu().reshape(xx.shape).numpy()
    
    plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap="coolwarm", edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

def plot_3d_surface(model, X, y, device):
    model.eval()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    grid = grid.to(device)

    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits)

    Z = probs.cpu().reshape(xx.shape).numpy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xx, yy, Z, alpha=0.6)
    
    ax.scatter(
        X[:, 0], X[:, 1], y.squeeze(),
        c=y.squeeze(), cmap='coolwarm', edgecolor='k'
    )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Probability")
    ax.set_title("3D Decision Surface")
    plt.show()
