import torch
from model import MyNeuralNet
from data import get_dataloaders
from utils import show_predictions, evaluate

def main():
    print("--- CIFAR-10 Inference Script ---")
    
    model_path = "cifar10_model_1.5.pth"
        
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Initializing ResNet9 architecture...")
    model = MyNeuralNet(num_classes=10)
    
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    model.to(device)
    print("Model successfully loaded and locked into evaluation mode.")
    
    print("\nLoading CIFAR-10 test data...")
    _, test_loader = get_dataloaders(batch_size=256)
    
    print("Running full evaluation on 10,000 test images (This might take a second)...")
    test_acc = evaluate(model, test_loader, device)
    print(f"--> Final Test Accuracy: {test_acc * 100:.2f}%")
    
    print("\nGenerating visual predictions...")
    print("A window will pop up showing the images. Close the window to exit the script.")
    show_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()
