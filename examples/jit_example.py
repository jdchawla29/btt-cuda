"""
Example demonstrating the JIT-compiled CUDA extension with BTT layers
"""
import sys
import os
import time
import torch

# Add the src directory to the path to import btt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.btt.reference import BTTLayer as CPUBTTLayer
from src.btt_cuda import BTTLayer as CUDABTTLayer
from src.btt_cuda import BTTLayerOptimized as CUDABTTLayerOptimized

# Parameters
batch_size = 32
d_in = 4096
d_out = 4096
tt_rank = 4
device = torch.device("cuda")

def main():
    # Create input tensor
    x = torch.randn(batch_size, d_in)
    x_cuda = x.to(device)
    
    # Initialize CPU layer
    print("Initializing CPU implementation...")
    cpu_layer = CPUBTTLayer(d_in, d_out, tt_rank)
    
    # Initialize CUDA layer
    print("Initializing basic CUDA implementation...")
    cuda_layer = CUDABTTLayer(d_in, d_out, tt_rank).to(device)
    
    # Initialize optimized CUDA layer
    print("Initializing optimized CUDA implementation...")
    cuda_opt_layer = CUDABTTLayerOptimized(d_in, d_out, tt_rank).to(device)
    
    # Copy weights to ensure fair comparison
    cuda_layer.W1.data = cpu_layer.W1.data.to(device)
    cuda_layer.W2.data = cpu_layer.W2.data.to(device)
    cuda_opt_layer.W1.data = cpu_layer.W1.data.to(device)
    cuda_opt_layer.W2.data = cpu_layer.W2.data.to(device)
    
    # Forward pass on CPU
    print("Running CPU forward pass...")
    start = time.time()
    cpu_out = cpu_layer(x)
    cpu_time = time.time() - start
    
    # Forward pass on CUDA
    print("Running basic CUDA forward pass...")
    start = time.time()
    cuda_out = cuda_layer(x_cuda).cpu()
    cuda_time = time.time() - start
    
    # Forward pass on optimized CUDA
    print("Running optimized CUDA forward pass...")
    start = time.time()
    cuda_opt_out = cuda_opt_layer(x_cuda).cpu()
    cuda_opt_time = time.time() - start
    
    # Verify outputs match
    cpu_cuda_diff = (cpu_out - cuda_out).abs().max().item()
    cpu_cuda_opt_diff = (cpu_out - cuda_opt_out).abs().max().item()
    
    print(f"\nResults:")
    print(f"CPU forward pass time: {cpu_time:.6f} seconds")
    print(f"Basic CUDA forward pass time: {cuda_time:.6f} seconds (speedup: {cpu_time/cuda_time:.2f}x)")
    print(f"Optimized CUDA forward pass time: {cuda_opt_time:.6f} seconds (speedup: {cpu_time/cuda_opt_time:.2f}x)")
    print(f"Max difference CPU vs basic CUDA: {cpu_cuda_diff:.6e}")
    print(f"Max difference CPU vs optimized CUDA: {cpu_cuda_opt_diff:.6e}")

if __name__ == "__main__":
    main()