import torch
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import List, Tuple, Dict, Any

try:
    from btt.reference import BTTLayer as PyBTTLayer
    from btt_cuda import BTTLayer as BTTCUDALayer
    from btt_cuda import BTTLayerOptimized as BTTCUDAOptimizedLayer
except ImportError:
    print("BTTLayerOptimized not found. Please install the package with `pip install .`")
    exit(1)

def time_forward_backward(
    layer: torch.nn.Module, 
    input_tensor: torch.Tensor,
    num_repeats: int = 10,
    warmup: int = 5
) -> Tuple[float, float]:
    """Measure forward and backward pass times."""
    device = next(layer.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    for _ in range(warmup):
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
    
    # Sync before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time forward pass
    forward_times = []
    for _ in range(num_repeats):
        start_time = time.time()
        output = layer(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)
    
    # Time backward pass
    backward_times = []
    for _ in range(num_repeats):
        output = layer(input_tensor)
        loss = output.sum()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        backward_times.append(time.time() - start_time)
    
    return np.mean(forward_times), np.mean(backward_times)


def run_benchmark(configs: List[Dict[str, Any]], num_repeats: int = 10) -> pd.DataFrame:
    """Run benchmark on multiple configurations."""
    results = []
    
    for config in configs:
        print(f"Running benchmark with {config}...")
        batch_size = config['batch_size']
        d_in = config['d_in']
        d_out = config['d_out']
        tt_rank = config['tt_rank']
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, d_in)
        
        # PyTorch CPU implementation
        py_layer = PyBTTLayer(d_in, d_out, tt_rank)
        py_fwd_time, py_bwd_time = time_forward_backward(py_layer, input_tensor, num_repeats)
        
        # CUDA implementation
        cuda_layer = BTTCUDALayer(d_in, d_out, tt_rank).cuda()
        # Copy weights from PyTorch implementation for fair comparison
        cuda_layer.W1.data = py_layer.W1.data.cuda()
        cuda_layer.W2.data = py_layer.W2.data.cuda()
        cuda_fwd_time, cuda_bwd_time = time_forward_backward(cuda_layer, input_tensor, num_repeats)
        
        # Optimized CUDA implementation
        opt_cuda_layer = BTTCUDAOptimizedLayer(d_in, d_out, tt_rank).cuda()
        # Copy weights from PyTorch implementation for fair comparison
        opt_cuda_layer.W1.data = py_layer.W1.data.cuda()
        opt_cuda_layer.W2.data = py_layer.W2.data.cuda()
        opt_cuda_fwd_time, opt_cuda_bwd_time = time_forward_backward(opt_cuda_layer, input_tensor, num_repeats)
        
        # Calculate speedups
        cuda_fwd_speedup = py_fwd_time / cuda_fwd_time
        cuda_bwd_speedup = py_bwd_time / cuda_bwd_time
        opt_fwd_speedup = py_fwd_time / opt_cuda_fwd_time
        opt_bwd_speedup = py_bwd_time / opt_cuda_bwd_time
        opt_vs_cuda_fwd_speedup = cuda_fwd_time / opt_cuda_fwd_time
        opt_vs_cuda_bwd_speedup = cuda_bwd_time / opt_cuda_bwd_time
        
        # Verify correctness
        with torch.no_grad():
            py_output = py_layer(input_tensor.cpu())
            cuda_output = cuda_layer(input_tensor.cuda()).cpu()
            opt_cuda_output = opt_cuda_layer(input_tensor.cuda()).cpu()
            cuda_diff = (py_output - cuda_output).abs().max().item()
            opt_cuda_diff = (py_output - opt_cuda_output).abs().max().item()
            opt_vs_cuda_diff = (cuda_output - opt_cuda_output).abs().max().item()
        
        # Collect results
        row = {
            'batch_size': batch_size,
            'd_in': d_in,
            'd_out': d_out,
            'tt_rank': tt_rank,
            'py_fwd_time': py_fwd_time,
            'py_bwd_time': py_bwd_time,
            'cuda_fwd_time': cuda_fwd_time,
            'cuda_bwd_time': cuda_bwd_time,
            'opt_cuda_fwd_time': opt_cuda_fwd_time,
            'opt_cuda_bwd_time': opt_cuda_bwd_time,
            'cuda_fwd_speedup': cuda_fwd_speedup,
            'cuda_bwd_speedup': cuda_bwd_speedup,
            'opt_fwd_speedup': opt_fwd_speedup,
            'opt_bwd_speedup': opt_bwd_speedup,
            'opt_vs_cuda_fwd_speedup': opt_vs_cuda_fwd_speedup,
            'opt_vs_cuda_bwd_speedup': opt_vs_cuda_bwd_speedup,
            'cuda_diff': cuda_diff,
            'opt_cuda_diff': opt_cuda_diff,
            'opt_vs_cuda_diff': opt_vs_cuda_diff
        }
        results.append(row)
    
    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_file: str = 'benchmark_results.png'):
    """Plot benchmark results."""
    plt.figure(figsize=(15, 10))
    
    # Forward pass speedup
    plt.subplot(2, 2, 1)
    x = range(len(df))
    plt.bar(x, df['cuda_fwd_speedup'], width=0.4, label='CUDA', alpha=0.7)
    plt.bar([i+0.4 for i in x], df['opt_fwd_speedup'], width=0.4, label='Optimized CUDA', alpha=0.7)
    plt.xticks([i+0.2 for i in x], [f"B:{r['batch_size']},In:{r['d_in']},Out:{r['d_out']},R:{r['tt_rank']}" for _, r in df.iterrows()], rotation=90)
    plt.ylabel('Speedup over CPU')
    plt.title('Forward Pass Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Backward pass speedup
    plt.subplot(2, 2, 2)
    plt.bar(x, df['cuda_bwd_speedup'], width=0.4, label='CUDA', alpha=0.7)
    plt.bar([i+0.4 for i in x], df['opt_bwd_speedup'], width=0.4, label='Optimized CUDA', alpha=0.7)
    plt.xticks([i+0.2 for i in x], [f"B:{r['batch_size']},In:{r['d_in']},Out:{r['d_out']},R:{r['tt_rank']}" for _, r in df.iterrows()], rotation=90)
    plt.ylabel('Speedup over CPU')
    plt.title('Backward Pass Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Optimized vs CUDA speedup
    plt.subplot(2, 2, 3)
    plt.bar(x, df['opt_vs_cuda_fwd_speedup'], width=0.4, label='Forward', alpha=0.7)
    plt.bar([i+0.4 for i in x], df['opt_vs_cuda_bwd_speedup'], width=0.4, label='Backward', alpha=0.7)
    plt.xticks([i+0.2 for i in x], [f"B:{r['batch_size']},In:{r['d_in']},Out:{r['d_out']},R:{r['tt_rank']}" for _, r in df.iterrows()], rotation=90)
    plt.ylabel('Speedup (Optimized / Original)')
    plt.title('Optimized CUDA Speedup over Original CUDA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Numerical differences
    plt.subplot(2, 2, 4)
    plt.bar(x, df['cuda_diff'], width=0.3, label='CUDA vs CPU', alpha=0.7)
    plt.bar([i+0.3 for i in x], df['opt_cuda_diff'], width=0.3, label='Optimized vs CPU', alpha=0.7)
    plt.bar([i+0.6 for i in x], df['opt_vs_cuda_diff'], width=0.3, label='Optimized vs CUDA', alpha=0.7)
    plt.xticks([i+0.3 for i in x], [f"B:{r['batch_size']},In:{r['d_in']},Out:{r['d_out']},R:{r['tt_rank']}" for _, r in df.iterrows()], rotation=90)
    plt.ylabel('Max Absolute Difference')
    plt.yscale('log')
    plt.title('Numerical Differences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results plotted to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark BTT implementations')
    parser.add_argument('--repeats', type=int, default=10, help='Number of runs to average')
    args = parser.parse_args()
    
    # Define benchmark configurations
    configs = [
        {'batch_size': 32, 'd_in': 1024, 'd_out': 1024, 'tt_rank': 16},
        {'batch_size': 64, 'd_in': 1024, 'd_out': 1024, 'tt_rank': 16},
        {'batch_size': 128, 'd_in': 1024, 'd_out': 1024, 'tt_rank': 16},
        {'batch_size': 64, 'd_in': 512, 'd_out': 512, 'tt_rank': 8},
        {'batch_size': 64, 'd_in': 2048, 'd_out': 2048, 'tt_rank': 32},
        {'batch_size': 64, 'd_in': 4096, 'd_out': 1024, 'tt_rank': 16},
        {'batch_size': 64, 'd_in': 1024, 'd_out': 4096, 'tt_rank': 16},
    ]
    
    # Run benchmark
    results = run_benchmark(configs, num_repeats=args.repeats)
    
    # Print results as a table
    table_headers = [
        'Config', 
        'CPU Fwd (s)', 'CPU Bwd (s)',
        'CUDA Fwd (s)', 'CUDA Bwd (s)',
        'Opt Fwd (s)', 'Opt Bwd (s)',
        'CUDA vs CPU Fwd', 'CUDA vs CPU Bwd',
        'Opt vs CPU Fwd', 'Opt vs CPU Bwd',
        'Opt vs CUDA Fwd', 'Opt vs CUDA Bwd',
        'Max Diff'
    ]
    table_rows = []
    for _, row in results.iterrows():
        config = f"B:{row['batch_size']}, In:{row['d_in']}, Out:{row['d_out']}, R:{row['tt_rank']}"
        table_row = [
            config,
            f"{row['py_fwd_time']:.6f}", f"{row['py_bwd_time']:.6f}",
            f"{row['cuda_fwd_time']:.6f}", f"{row['cuda_bwd_time']:.6f}",
            f"{row['opt_cuda_fwd_time']:.6f}", f"{row['opt_cuda_bwd_time']:.6f}",
            f"{row['cuda_fwd_speedup']:.2f}x", f"{row['cuda_bwd_speedup']:.2f}x",
            f"{row['opt_fwd_speedup']:.2f}x", f"{row['opt_bwd_speedup']:.2f}x",
            f"{row['opt_vs_cuda_fwd_speedup']:.2f}x", f"{row['opt_vs_cuda_bwd_speedup']:.2f}x",
            f"{row['opt_vs_cuda_diff']:.2e}"
        ]
        table_rows.append(table_row)
    
    print("\nBenchmark Results:")
    print(tabulate(table_rows, headers=table_headers, tablefmt="fancy_grid"))
    
    # Plot results
    plot_results(results)
    
    # Save results to CSV
    results.to_csv('benchmark_results.csv', index=False)
    print("Results saved to benchmark_results.csv")


if __name__ == "__main__":
    main()