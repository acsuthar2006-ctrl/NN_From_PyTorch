import torch
import torch.nn as nn
from data import get_dataloaders
from model import MyNeuralNet
from utils import evaluate, plot_decision_boundary, plot_3d_surface

def main():
    # 1. Setup Device
    # We can use MPS on mac if available, otherwise fallback to CPU. 
    # Your notebook had device="cpu", so we'll start with that but make it flexible!
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = 'cpu'
    print(f"Using device: {device}")

    # 2. Get Data
    print("Loading data...")
    train_loader, test_loader, X_full, y_full = get_dataloaders(
        n_samples=1000, noise=0.2, random_state=42, batch_size=128, use_complex=True
    )

    # 3. Setup Model
    model = MyNeuralNet(25, [2, 64, 1])
    model.init_weights()
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = 2000

    # 4. Training Loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
            
        train_loss /= len(train_loader)
        
        # Print progress every 100 epochs instead of 50 to keep output clean
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {train_loss:.4f}")

    # 5. Evaluation
    print("\n--- Evaluation ---")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"Train Acc: {train_acc * 100:.2f}%")
    print(f"Test Acc:  {test_acc * 100:.2f}%")

    # 6. Visualization
    print("\nGenerating plots...")
    # Convert the full tensor data back to numpy for plotting
    X_np = X_full.numpy()
    y_np = y_full.numpy()
    
    plot_decision_boundary(model, X_np, y_np, device)
    plot_3d_surface(model, X_np, y_np, device)

if __name__ == "__main__":
    main()
