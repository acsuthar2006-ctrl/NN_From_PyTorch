import torch
import os
import sys
from model import MyNeuralNet
from data import get_dataloaders
from utils import show_predictions, evaluate

def main():
    print("--- CIFAR-10 Inference Script ---")
    
    # 1. Check if the model file exists
    model_path = "cifar10_model.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find '{model_path}'.")
        print("Please download it from Kaggle and place it in this folder.")
        sys.exit(1)
        
    # 2. Setup Device (MPS for Mac, CUDA for Nvidia, CPU fallback)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 3. Instantiate a Blank Model (ResNet9)
    print("Initializing ResNet9 architecture...")
    model = MyNeuralNet(num_classes=10)
    
    # 4. Load the Trained Brain!
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # CRITICAL: Put model in evaluation mode (disables Dropout and locks BatchNorm)
    model.eval()
    model.to(device)
    print("Model successfully loaded and locked into evaluation mode.")
    
    # 5. Load the Test Data
    print("\nLoading CIFAR-10 test data...")
    # We don't need train_loader, just test_loader
    _, test_loader = get_dataloaders(batch_size=256)
    
    # 6. Run Accuracy Evaluation
    print("Running full evaluation on 10,000 test images (This might take a second)...")
    test_acc = evaluate(model, test_loader, device)
    print(f"--> Final Test Accuracy: {test_acc * 100:.2f}%")
    
    # 7. Visual Predictions
    print("\nGenerating visual predictions...")
    print("A window will pop up showing the images. Close the window to exit the script.")
    show_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()
