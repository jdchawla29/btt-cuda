import sys
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from torch.profiler import profile, record_function, ProfilerActivity

# Add the src directory to the path to import btt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.btt.reference import BTTLayer as CPUBTTLayer
from src.btt_cuda import BTTLayer as CUDABTTLayer
from src.btt_cuda import BTTLayerOptimized as CUDABTTLayerOptimized

# Default parameters
DEFAULT_CONFIG = {
    "batch_size": 32,
    "d_in": 1024,
    "d_out": 512,
    "tt_rank": 4,
}

def benchmark_forward(model, inputs, num_warmup=5, num_iter=20):
    """Run benchmark with warmup and multiple iterations"""
    # Warmup
    for _ in range(num_warmup):
        _ = model(inputs)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iter):
        _ = model(inputs)
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / num_iter

def benchmark_backward(model, inputs, num_warmup=5, num_iter=20):
    """Run backward pass benchmark with warmup and multiple iterations"""
    # Warmup
    for _ in range(num_warmup):
        out = model(inputs)
        loss = out.sum()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iter):
        out = model(inputs)
        loss = out.sum()
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / num_iter

def profile_model(model_type, batch_size, d_in, d_out, tt_rank, device="cuda"):
    """Profile a specific model configuration"""
    print(f"\nProfiling {model_type} with batch_size={batch_size}, d_in={d_in}, d_out={d_out}, tt_rank={tt_rank}")
    
    # Create input tensor
    x = torch.randn(batch_size, d_in)
    x_cuda = x.to(device) if device == "cuda" else x
    
    if model_type == "cpu":
        model = CPUBTTLayer(d_in, d_out, tt_rank)
        inputs = x
    elif model_type == "cuda_basic":
        model = CUDABTTLayer(d_in, d_out, tt_rank).to(device)
        inputs = x_cuda
    elif model_type == "cuda_optimized":
        model = CUDABTTLayerOptimized(d_in, d_out, tt_rank).to(device)
        inputs = x_cuda
    
    # Memory usage before inference (CUDA only)
    if device == "cuda" and model_type != "cpu":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Time forward pass
    forward_time = benchmark_forward(model, inputs)
    
    # Memory usage after inference (CUDA only)
    if device == "cuda" and model_type != "cpu":
        torch.cuda.synchronize()
        mem_after = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        mem_usage = mem_after - mem_before
    else:
        mem_usage = 0
    
    # Time backward pass
    backward_time = benchmark_backward(model, inputs)
    
    # Detailed profiling with PyTorch profiler (CUDA only)
    if device == "cuda" and model_type != "cpu":
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with record_function("model_inference"):
                out = model(inputs)
                out.sum().backward()
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return {
        "model_type": model_type,
        "batch_size": batch_size,
        "d_in": d_in,
        "d_out": d_out,
        "tt_rank": tt_rank,
        "forward_time": forward_time,
        "backward_time": backward_time,
        "memory_usage_mb": mem_usage,
    }

def run_comprehensive_profiling():
    """Run profiling across different configurations"""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Different batch sizes
    for batch_size in [1, 8, 32, 64, 128]:
        for model_type in ["cpu", "cuda_optimized"]:
            if model_type == "cpu" and batch_size > 32:
                continue  # Skip large batch sizes for CPU to save time
                
            result = profile_model(
                model_type, 
                batch_size, 
                DEFAULT_CONFIG["d_in"], 
                DEFAULT_CONFIG["d_out"], 
                DEFAULT_CONFIG["tt_rank"],
                device=device if model_type != "cpu" else "cpu"
            )
            results.append(result)
    
    # Different dimensions
    for d_size in [256, 512, 1024, 2048]:
        for model_type in ["cuda_basic", "cuda_optimized"]:  # Skip CPU for large dims
            result = profile_model(
                model_type, 
                DEFAULT_CONFIG["batch_size"], 
                d_size, 
                d_size // 2, 
                DEFAULT_CONFIG["tt_rank"],
                device=device
            )
            results.append(result)
    
    # Different TT ranks
    for rank in [8, 16, 32, 64]:
        for model_type in ["cuda_basic", "cuda_optimized"]:  # Skip CPU for high ranks
            result = profile_model(
                model_type, 
                DEFAULT_CONFIG["batch_size"], 
                DEFAULT_CONFIG["d_in"], 
                DEFAULT_CONFIG["d_out"], 
                rank,
                device=device
            )
            results.append(result)
    
    return results

def visualize_results(results):
    """Create visualizations for profiling results"""
    df = pd.DataFrame(results)
    
    # Plot 1: Batch size vs. Forward time
    plt.figure(figsize=(10, 6))
    for model in df['model_type'].unique():
        subset = df[(df['model_type'] == model) & 
                   (df['d_in'] == DEFAULT_CONFIG['d_in']) & 
                   (df['tt_rank'] == DEFAULT_CONFIG['tt_rank'])]
        if not subset.empty:
            plt.plot(subset['batch_size'], subset['forward_time'], 'o-', label=model)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Forward Time (s)')
    plt.title('Forward Pass Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('batch_size_scaling.png')
    
    # Plot 2: Input dimension vs. Forward time
    plt.figure(figsize=(10, 6))
    for model in ['cuda_basic', 'cuda_optimized']:
        subset = df[(df['model_type'] == model) & 
                   (df['batch_size'] == DEFAULT_CONFIG['batch_size'])]
        subset = subset[subset['d_in'] == subset['d_out'] * 2]  # Filter for dimension scaling
        if not subset.empty:
            plt.plot(subset['d_in'], subset['forward_time'], 'o-', label=model)
    
    plt.xlabel('Input Dimension')
    plt.ylabel('Forward Time (s)')
    plt.title('Forward Pass Time vs Input Dimension')
    plt.legend()
    plt.grid(True)
    plt.savefig('dimension_scaling.png')
    
    # Plot 3: Rank vs Memory Usage
    plt.figure(figsize=(10, 6))
    for model in ['cuda_basic', 'cuda_optimized']:
        subset = df[(df['model_type'] == model) & 
                   (df['batch_size'] == DEFAULT_CONFIG['batch_size']) &
                   (df['d_in'] == DEFAULT_CONFIG['d_in'])]
        if not subset.empty:
            plt.plot(subset['tt_rank'], subset['memory_usage_mb'], 'o-', label=model)
    
    plt.xlabel('TT Rank')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs TT Rank')
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_usage.png')
    
    # Save raw data
    df.to_csv('profiling_results.csv', index=False)
    print(f"Results saved to profiling_results.csv")
    print(f"Plots saved as PNG files")

def main():
    """Main function to run profiling"""
    print("Starting comprehensive profiling of BTT layers...")
    
    # Print device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run profiling
    results = run_comprehensive_profiling()
    
    # Visualize
    visualize_results(results)

if __name__ == "__main__":
    main()