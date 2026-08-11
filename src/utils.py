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
            
            # Flatten the image for MLP
            X_batch_flat = X_batch.view(X_batch.size(0), -1)
            
            logits = model(X_batch_flat)
            # For multi-class, prediction is the index with the highest logit
            preds = logits.argmax(dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total

def show_predictions(model, loader, device, num_images=16):
    """Visualizes a grid of test images along with their predicted and true labels."""
    model.eval()
    
    # Get a single batch of test data
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Select the first 'num_images'
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Move to device and get predictions
    images_gpu = images.to(device)
    # Flatten for the MLP
    images_flat = images_gpu.view(images_gpu.size(0), -1)
    
    with torch.no_grad():
        logits = model(images_flat)
        preds = logits.argmax(dim=1).cpu()

    # Create a plot
    fig = plt.figure(figsize=(8, 8))
    
    for i in range(num_images):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        
        # Un-normalize the image for display
        # We used mean=0.1307, std=0.3081 during data loading
        img = images[i].squeeze().numpy()
        img = (img * 0.3081) + 0.1307
        img = np.clip(img, 0, 1)
        
        ax.imshow(img, cmap='gray')
        
        # Set title color: green if correct, red if wrong
        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {preds[i].item()} | True: {labels[i].item()}", color=color)
        
    plt.tight_layout()
    plt.show()
