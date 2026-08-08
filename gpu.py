import torch

# Check if Apple Silicon GPU (MPS) is available
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    
    # Create two random tensors directly on the Mac GPU
    x = torch.randn(1000, 1000, device=mps_device)
    y = torch.randn(1000, 1000, device=mps_device)
    
    # Perform a matrix multiplication on the GPU
    result = torch.matmul(x, y)
    
    print("SUCCESS: Your Mac GPU (MPS) is active and working!")
    print(f"Result shape calculated on GPU: {result.shape}")
else:
    print("FAILED: Mac GPU (MPS) is NOT available. Running on CPU.")
    
