import torch
from btt import BTTLayerCUDA

# Basic usage
def basic_example():
    # Create BTT layer - input dim 64, output dim 64, rank 8
    layer = BTTLayerCUDA(d_in=64, d_out=64, tt_rank=2).cuda()
    
    # Create random input
    batch_size = 32
    x = torch.randn(batch_size, 64).cuda()
    
    # Forward pass
    out = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

# Compare with CPU version
def compare_cpu_gpu():
    from btt import BTTLayer
    
    # Create both CPU and CUDA layers
    layer_cpu = BTTLayer(64, 64, tt_rank=8)
    layer_cuda = BTTLayerCUDA(64, 64, tt_rank=8).cuda()
    
    # Copy weights to ensure identical initialization
    layer_cuda.W1.data = layer_cpu.W1.data.cuda()
    layer_cuda.W2.data = layer_cpu.W2.data.cuda()
    
    # Test input
    x = torch.randn(32, 64)
    out_cpu = layer_cpu(x)
    out_cuda = layer_cuda(x.cuda()).cpu()
    
    # Check results match
    max_diff = (out_cpu - out_cuda).abs().max().item()
    print(f"Max difference between CPU and GPU: {max_diff}")

# Gradient checking
def check_gradients():
    layer = BTTLayerCUDA(64, 64, tt_rank=8).cuda()
    x = torch.randn(32, 64, requires_grad=True).cuda()
    
    # Forward and backward pass
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    print("Gradient shapes:")
    print(f"Input grad: {x.grad.shape}")
    print(f"W1 grad: {layer.W1.grad.shape}")
    print(f"W2 grad: {layer.W2.grad.shape}")

# Performance benchmark
def benchmark():
    layer = BTTLayerCUDA(1024, 1024, tt_rank=8).cuda()
    x = torch.randn(128, 1024).cuda()
    
    # Warmup
    for _ in range(10):
        out = layer(x)
    
    # Time forward passes
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    num_iters = 100
    for _ in range(num_iters):
        out = layer(x)
    end.record()
    
    torch.cuda.synchronize()
    time_per_iter = start.elapsed_time(end) / num_iters
    print(f"Average time per forward pass: {time_per_iter:.3f} ms")

# # Use in a larger model
# class MLPWithBTT(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = BTTLayerCUDA(784, 512, tt_rank=8)  # For MNIST
#         self.relu = torch.nn.ReLU()
#         self.layer2 = BTTLayerCUDA(512, 10, tt_rank=8)
    
#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = self.relu(self.layer1(x))
#         return self.layer2(x)

if __name__ == "__main__":
    print("Basic example:")
    basic_example()
    print("\nCPU vs GPU comparison:")
    compare_cpu_gpu()
    print("\nGradient checking:")
    check_gradients()
    print("\nBenchmark:")
    benchmark()