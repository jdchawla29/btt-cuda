import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch.profiler import profile, ProfilerActivity
import pandas as pd
from typing import List, Tuple, Dict

from btt import BTTLayer as PyBTTLayer
from btt_cuda import BTTLayer

class BTTBenchmark:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def benchmark_shapes(self) -> List[Dict]:
        """Benchmark different input/output shapes and ranks"""
        results = []
        batch_sizes = [32, 64, 128, 256]
        input_dims = [64, 256, 1024, 4096] # 8x8, 16x16, 32x32, 64x64 
        ranks = [4, 8, 16, 32]

        for batch_size in batch_sizes:
            for d_in in input_dims:
                for rank in ranks:
                    d_out = d_in * 4  # Example scaling
                    
                    # Warm up
                    self._run_single_benchmark(batch_size, d_in, d_out, rank)
                    
                    # Actual timing
                    cuda_time, ref_time, cuda_mem, ref_mem = self._run_single_benchmark(
                        batch_size, d_in, d_out, rank
                    )
                    
                    results.append({
                        'batch_size': batch_size,
                        'd_in': d_in,
                        'd_out': d_out,
                        'rank': rank,
                        'cuda_time': cuda_time,
                        'ref_time': ref_time,
                        'speedup': ref_time/cuda_time,
                        'cuda_mem': cuda_mem,
                        'ref_mem': ref_mem
                    })
                    
        return results

    def _run_single_benchmark(self, batch_size: int, d_in: int, d_out: int, rank: int) -> Tuple[float, float, float, float]:
        """Run a single benchmark configuration"""
        # Reference implementation
        ref_layer = PyBTTLayer(d_in, d_out, rank).to(self.device)
        x = torch.randn(batch_size, d_in, device=self.device)
        
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        y_ref = ref_layer(x)
        ref_time = time.perf_counter() - start
        ref_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # CUDA implementation  
        cuda_layer = BTTLayer(d_in, d_out, rank).to(self.device)
        
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        y_cuda = cuda_layer(x)
        cuda_time = time.perf_counter() - start
        cuda_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return cuda_time, ref_time, cuda_mem, ref_mem

    def plot_results(self, results: List[Dict]):
        """Create visualizations of benchmark results"""
        df = pd.DataFrame(results)
        
        # 1. Basic line plot with different lines for (batch_size, rank) combinations
        plt.figure(figsize=(12, 8))
        for batch in sorted(df['batch_size'].unique()):
            for rank in sorted(df['rank'].unique()):
                subset = df[(df['batch_size'] == batch) & (df['rank'] == rank)]
                plt.plot(subset['d_in'], subset['speedup'], 
                        marker='o', 
                        label=f'Batch={batch}, Rank={rank}')
        
        plt.xlabel('Input Dimension')
        plt.ylabel('Speedup (Reference/CUDA)')
        plt.title('BTT CUDA Speedup vs Input Size')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('speedup_vs_size.png')
        plt.close()

        # 2. Multiple heatmaps for different ranks
        ranks = sorted(df['rank'].unique())
        n_ranks = len(ranks)
        fig, axes = plt.subplots(1, n_ranks, figsize=(5*n_ranks, 6))
        
        for idx, rank in enumerate(ranks):
            rank_data = df[df['rank'] == rank]
            pivot = rank_data.pivot_table(
                values='speedup',
                index='batch_size',
                columns='d_in',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[idx])
            axes[idx].set_title(f'Rank = {rank}')
            axes[idx].set_xlabel('Input Dimension')
            if idx == 0:
                axes[idx].set_ylabel('Batch Size')
            else:
                axes[idx].set_ylabel('')
        
        plt.suptitle('Speedup Heatmaps Across Different Ranks')
        plt.tight_layout()
        plt.savefig('speedup_heatmaps.png')
        plt.close()

        # 3. 3D scatter plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df['batch_size'], 
                            df['d_in'], 
                            df['rank'],
                            c=df['speedup'],
                            cmap='viridis')
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Input Dimension')
        ax.set_zlabel('Rank')
        plt.colorbar(scatter, label='Speedup (Reference/CUDA)')
        plt.title('Speedup Distribution in Parameter Space')
        plt.tight_layout()
        plt.savefig('speedup_3d.png')
        plt.close()

        # 4. Summary statistics
        summary_stats = df.groupby(['batch_size', 'd_in', 'rank'])['speedup'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        print("\nTop 10 Configurations by Speedup:")
        print(summary_stats.nlargest(10, 'mean')[
            ['batch_size', 'd_in', 'rank', 'mean', 'std'
        ]])
        
        # Save detailed statistics
        summary_stats.to_csv('speedup_statistics.csv', index=False)

def main():
    benchmark = BTTBenchmark()
    
    # Run comprehensive benchmarks
    results = benchmark.benchmark_shapes()
    
    # Create visualizations
    benchmark.plot_results(results)
    
    # Save results to CSV
    pd.DataFrame(results).to_csv('btt_benchmark_results.csv', index=False)

if __name__ == "__main__":
    main()