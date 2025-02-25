import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Import layers from different implementations
from btt.reference import BTTLayer as PyBTTLayer
from btt_cuda import BTTLayer as CUDABTTLayer
try:
    from btt_cuda import BTTLayerOptimized as OptCUDABTTLayer
    OPTIMIZED_AVAILABLE = True
except ImportError:
    print("Optimized implementation not available, please run 'pip install .'")
    OPTIMIZED_AVAILABLE = False

def time_execution(func, n_runs=10):
    """Measure execution time of a function."""
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()  # Ensure GPU operations are completed
        start = time.time()
        func()
        torch.cuda.synchronize()
        times.append(time.time() - start)
    return np.mean(times), np.std(times)

def main():
    # Define model parameters
    batch_size = 64
    d_in = 2048
    d_out = 2048
    tt_rank = 16
    
    # Create input tensor
    x = torch.randn(batch_size, d_in)
    x_cuda = x.cuda()
    
    # Create layers for each implementation
    print(f"\nCreating BTT Layers (d_in={d_in}, d_out={d_out}, rank={tt_rank})...")
    py_layer = PyBTTLayer(d_in, d_out, tt_rank)
    cuda_layer = CUDABTTLayer(d_in, d_out, tt_rank).cuda()
    
    # Copy weights for fair comparison
    cuda_layer.W1.data = py_layer.W1.data.cuda()
    cuda_layer.W2.data = py_layer.W2.data.cuda()
    
    if OPTIMIZED_AVAILABLE:
        opt_layer = OptCUDABTTLayer(d_in, d_out, tt_rank).cuda()
        opt_layer.W1.data = py_layer.W1.data.cuda()
        opt_layer.W2.data = py_layer.W2.data.cuda()
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        py_layer(x)
        cuda_layer(x_cuda)
        if OPTIMIZED_AVAILABLE:
            opt_layer(x_cuda)
    
    # Measure forward pass times
    print("Measuring forward pass times...")
    py_fwd_time, py_fwd_std = time_execution(lambda: py_layer(x))
    cuda_fwd_time, cuda_fwd_std = time_execution(lambda: cuda_layer(x_cuda))
    if OPTIMIZED_AVAILABLE:
        opt_fwd_time, opt_fwd_std = time_execution(lambda: opt_layer(x_cuda))
    
    # Measure backward pass times
    print("Measuring backward pass times...")
    py_output = py_layer(x)
    py_loss = py_output.sum()
    py_bwd_time, py_bwd_std = time_execution(lambda: py_loss.backward(retain_graph=True))
    
    cuda_output = cuda_layer(x_cuda)
    cuda_loss = cuda_output.sum()
    cuda_bwd_time, cuda_bwd_std = time_execution(lambda: cuda_loss.backward(retain_graph=True))
    
    if OPTIMIZED_AVAILABLE:
        opt_output = opt_layer(x_cuda)
        opt_loss = opt_output.sum()
        opt_bwd_time, opt_bwd_std = time_execution(lambda: opt_loss.backward(retain_graph=True))
    
    # Check numerical differences
    if OPTIMIZED_AVAILABLE:
        with torch.no_grad():
            cuda_vs_py_diff = (py_output - cuda_output.cpu()).abs().max().item()
            opt_vs_py_diff = (py_output - opt_output.cpu()).abs().max().item()
            opt_vs_cuda_diff = (cuda_output - opt_output).abs().max().item()
    
    # Print results
    headers = ["Implementation", "Forward (ms)", "Backward (ms)", "Total (ms)"]
    rows = [
        ["PyTorch CPU", f"{py_fwd_time*1000:.3f} ± {py_fwd_std*1000:.3f}", f"{py_bwd_time*1000:.3f} ± {py_bwd_std*1000:.3f}", f"{(py_fwd_time+py_bwd_time)*1000:.3f}"],
        ["CUDA", f"{cuda_fwd_time*1000:.3f} ± {cuda_fwd_std*1000:.3f}", f"{cuda_bwd_time*1000:.3f} ± {cuda_bwd_std*1000:.3f}", f"{(cuda_fwd_time+cuda_bwd_time)*1000:.3f}"],
    ]
    
    if OPTIMIZED_AVAILABLE:
        rows.append(["CUDA Optimized", f"{opt_fwd_time*1000:.3f} ± {opt_fwd_std*1000:.3f}", f"{opt_bwd_time*1000:.3f} ± {opt_bwd_std*1000:.3f}", f"{(opt_fwd_time+opt_bwd_time)*1000:.3f}"])
    
    print("\nPerformance Comparison:")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    
    # Print speedups
    print("\nSpeedup Factors:")
    print(f"CUDA vs CPU (forward): {py_fwd_time/cuda_fwd_time:.2f}x")
    print(f"CUDA vs CPU (backward): {py_bwd_time/cuda_bwd_time:.2f}x")
    
    if OPTIMIZED_AVAILABLE:
        print(f"Optimized vs CPU (forward): {py_fwd_time/opt_fwd_time:.2f}x")
        print(f"Optimized vs CPU (backward): {py_bwd_time/opt_bwd_time:.2f}x")
        print(f"Optimized vs CUDA (forward): {cuda_fwd_time/opt_fwd_time:.2f}x")
        print(f"Optimized vs CUDA (backward): {cuda_bwd_time/opt_bwd_time:.2f}x")
        
        print("\nNumerical Precision:")
        print(f"CUDA vs CPU max difference: {cuda_vs_py_diff:.2e}")
        print(f"Optimized vs CPU max difference: {opt_vs_py_diff:.2e}")
        print(f"Optimized vs CUDA max difference: {opt_vs_cuda_diff:.2e}")
    
    # Plot results
    if OPTIMIZED_AVAILABLE:
        plt.figure(figsize=(10, 6))
        implementations = ["PyTorch CPU", "CUDA", "CUDA Optimized"]
        forward_times = [py_fwd_time*1000, cuda_fwd_time*1000, opt_fwd_time*1000]
        backward_times = [py_bwd_time*1000, cuda_bwd_time*1000, opt_bwd_time*1000]
        
        x = np.arange(len(implementations))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, forward_times, width, label='Forward')
        rects2 = ax.bar(x + width/2, backward_times, width, label='Backward')
        
        ax.set_ylabel('Time (ms)')
        ax.set_title('BTT Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(implementations)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('btt_performance_comparison.png')
        print("\nPlot saved to 'btt_performance_comparison.png'")

if __name__ == "__main__":
    main()