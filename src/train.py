import torch
import torch.nn as nn
from data import get_dataloaders
from model import MyNeuralNet
from utils import evaluate

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, test_loader = get_dataloaders()

    model = MyNeuralNet(num_classes=10)
    model.init_weights()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # [NEW] Add weight_decay (L2 regularization) to AdamW to stop overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    epochs = 50
    
    # 1. Learning Rate Scheduler (OneCycleLR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)
    
    # 2. Automatic Mixed Precision (AMP) Scaler for fast Kaggle GPU training
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    # 4. Training Loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Flatten images for the MLP
            # batch_X_flat = batch_X.view(batch_X.size(0), -1)

            optimizer.zero_grad()
            
            # 3. AMP Training Step (Only runs if on CUDA/Kaggle)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            # Step the scheduler after EACH BATCH (required for OneCycleLR)
            scheduler.step()
            
            train_loss += loss.item() 
            
        train_loss /= len(train_loader)
        
        # Print progress and current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | LR: {current_lr:.6f}")

    # 5. Evaluation
    print("\n--- Evaluation ---")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Model now handles 3D images directly
            # batch_X_flat = batch_X.view(batch_X.size(0), -1)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"Train Acc: {train_acc * 100:.2f}%")
    print(f"Test Acc:  {test_acc * 100:.2f}%")

    # 6. Save the Model
    print("\nSaving the model parameters...")
    torch.save(model.state_dict(), "cifar10_model.pth")
    print("Model saved to 'cifar10_model.pth'!")

    # 7. Visualization
    print("\nGenerating predictions visualization...")
    from utils import show_predictions
    show_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()
