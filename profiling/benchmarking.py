import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from math import sqrt, isqrt

from btt import BTTLayer as PyBTTLayer
from btt_cuda import BTTLayer

class BTTPrecisionBenchmark:
    def __init__(self, rtol=1e-5, atol=1e-8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rtol = rtol  # Relative tolerance
        self.atol = atol  # Absolute tolerance
        
    def is_perfect_square(self, n: int) -> bool:
        """Check if a number is a perfect square"""
        root = isqrt(n)
        return root * root == n
        
    def generate_square_dimensions(self, min_dim: int, max_dim: int) -> List[int]:
        """Generate a list of perfect square dimensions between min_dim and max_dim"""
        dimensions = []
        current = isqrt(min_dim)
        while current * current <= max_dim:
            dimensions.append(current * current)
            current += 1
        return dimensions
        
    def generate_test_configurations(self) -> List[Tuple[int, int, int, int]]:
        """Generate test configurations (batch_size, d_in, d_out, rank), targeting ~5000 cases"""
        # Use fewer batch sizes - most precision issues won't be batch-dependent
        batch_sizes = [32, 128, 512]
        
        # Generate square input dimensions with some spacing
        all_input_dims = self.generate_square_dimensions(16, 4096)  # 4x4 to 64x64
        input_dims = [d for i, d in enumerate(all_input_dims) if i % 2 == 0]  # Take every other dimension
        
        # Output dimensions starting from 1, then selected perfect squares
        all_output_dims = [1]  # Start with 1x1
        all_output_dims.extend(self.generate_square_dimensions(4, 16384))
        # Take every other dimension, but ensure we keep some key ratios relative to each input dim
        output_dims = set([1])  # Always include 1
        for d_in in input_dims:
            # Include d_out = 1
            output_dims.add(1)
            # Include same dimension as input
            if d_in <= 16384:
                output_dims.add(d_in)
            # Include half and double if they're perfect squares
            half = d_in // 2
            if self.is_perfect_square(half):
                output_dims.add(half)
            double = d_in * 2
            if double <= 16384 and self.is_perfect_square(double):
                output_dims.add(double)
        output_dims = sorted(list(output_dims))
        
        # Use fewer ranks
        ranks = [4, 16, 64]
        
        configs = []
        for batch_size in batch_sizes:
            for d_in in input_dims:
                for d_out in output_dims:
                    # Only constrain maximum output size relative to input
                    if d_out > d_in * 16:
                        continue
                        
                    for rank in ranks:
                        # Adjust rank constraint for very small output dimensions
                        max_rank = min(d_in, d_out) if d_out > 1 else d_in
                        if rank > max_rank / 2:
                            continue
                            
                        configs.append((batch_size, d_in, d_out, rank))
        
        return configs

    def _test_single_configuration(self, batch_size: int, d_in: int, d_out: int, rank: int) -> Dict:
        """Test a single configuration for numerical precision"""
        # Initialize layers
        ref_layer = PyBTTLayer(d_in, d_out, rank).to(self.device)
        cuda_layer = BTTLayer(d_in, d_out, rank).to(self.device)
        
        # Ensure both layers have the same weights
        cuda_layer.load_state_dict(ref_layer.state_dict())
        
        # Generate input
        torch.manual_seed(42)  # For reproducibility
        x = torch.randn(batch_size, d_in, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            y_ref = ref_layer(x)
            y_cuda = cuda_layer(x)
        
        # Compute differences
        abs_diff = torch.abs(y_cuda - y_ref)
        rel_diff = abs_diff / (torch.abs(y_ref) + self.atol)
        
        max_abs_diff = float(torch.max(abs_diff))
        max_rel_diff = float(torch.max(rel_diff))
        mean_abs_diff = float(torch.mean(abs_diff))
        std_abs_diff = float(torch.std(abs_diff))
        
        # Check if within tolerance
        passed = torch.allclose(y_cuda, y_ref, rtol=self.rtol, atol=self.atol)
        
        return {
            'batch_size': batch_size,
            'd_in': d_in,
            'd_out': d_out,
            'rank': rank,
            'max_abs_diff': max_abs_diff,
            'max_rel_diff': max_rel_diff,
            'mean_abs_diff': mean_abs_diff,
            'std_abs_diff': std_abs_diff,
            'passed': passed,
            'error': None
        }
        
    def benchmark_precision(self) -> List[Dict]:
        """Benchmark precision across different configurations"""
        results = []
        configs = self.generate_test_configurations()
        
        print(f"Testing {len(configs)} configurations")
        print("\nSample configurations:")
        for config in configs[:5]:
            print(f"batch={config[0]}, d_in={config[1]}, d_out={config[2]}, rank={config[3]}")
        print("...\n")
        
        for idx, (batch_size, d_in, d_out, rank) in enumerate(configs, 1):
            print(f"\rTesting configuration {idx}/{len(configs)}", end="")
            
            try:
                result = self._test_single_configuration(
                    batch_size, d_in, d_out, rank
                )
                results.append(result)
                
            except Exception as e:
                print(f"\nError in configuration: batch={batch_size}, d_in={d_in}, d_out={d_out}, rank={rank}")
                print(f"Error message: {str(e)}")
                results.append({
                    'batch_size': batch_size,
                    'd_in': d_in,
                    'd_out': d_out,
                    'rank': rank,
                    'max_abs_diff': float('nan'),
                    'max_rel_diff': float('nan'),
                    'mean_abs_diff': float('nan'),
                    'std_abs_diff': float('nan'),
                    'passed': False,
                    'error': str(e)
                })
        
        print("\nBenchmarking completed!")
        return results

    def summarize_results(self, results: List[Dict]):
        """Summarize benchmark results"""
        df = pd.DataFrame(results)
        
        # Overall statistics
        total_configs = len(df)
        passed_configs = df['passed'].sum()
        failed_configs = total_configs - passed_configs
        
        print("\nOverall Results:")
        print(f"Total configurations tested: {total_configs}")
        print(f"Passed: {passed_configs} ({passed_configs/total_configs*100:.2f}%)")
        print(f"Failed: {failed_configs} ({failed_configs/total_configs*100:.2f}%)")
        
        # Special analysis for d_out = 1 cases
        d_out_1_cases = df[df['d_out'] == 1]
        if not d_out_1_cases.empty:
            print("\nResults for d_out = 1 cases:")
            print(f"Total cases: {len(d_out_1_cases)}")
            print(f"Passed: {d_out_1_cases['passed'].sum()} ({d_out_1_cases['passed'].mean()*100:.2f}%)")
        
        # Group results by dimension ratios (excluding d_out = 1)
        df_ratio = df[df['d_out'] > 1].copy()
        if not df_ratio.empty:
            df_ratio['dim_ratio'] = df_ratio['d_out'] / df_ratio['d_in']
            ratio_stats = df_ratio.groupby(pd.qcut(df_ratio['dim_ratio'], 5))['passed'].agg(['count', 'mean'])
            print("\nResults by dimension ratio (d_out/d_in) [excluding d_out = 1]:")
            print(ratio_stats)
        
        # Show failed configurations
        if failed_configs > 0:
            print("\nFailed Configurations:")
            failed_df = df[~df['passed']].sort_values('max_abs_diff', ascending=False)
            print(failed_df[['batch_size', 'd_in', 'd_out', 'rank', 'max_abs_diff', 'max_rel_diff', 'error']])
        
        # Statistics for passed configurations
        passed_df = df[df['passed']]
        if not passed_df.empty:
            print("\nStatistics for Passed Configurations:")
            print("Maximum absolute difference statistics:")
            print(passed_df['max_abs_diff'].describe())
            print("\nMaximum relative difference statistics:")
            print(passed_df['max_rel_diff'].describe())
        
        # Save results
        df.to_csv('btt_precision_results.csv', index=False)
        
        # Save worst performing cases
        if not passed_df.empty:
            # Separate analysis for d_out = 1 and d_out > 1
            for d_out_case in [1, '>1']:
                if d_out_case == 1:
                    case_df = passed_df[passed_df['d_out'] == 1]
                    suffix = 'dout_1'
                else:
                    case_df = passed_df[passed_df['d_out'] > 1]
                    suffix = 'dout_gt_1'
                
                if not case_df.empty:
                    print(f"\nTop 10 worst performing passing configurations (d_out {d_out_case}):")
                    worst_passing = case_df.nlargest(10, 'max_rel_diff')[
                        ['batch_size', 'd_in', 'd_out', 'rank', 'max_abs_diff', 'max_rel_diff', 'mean_abs_diff']
                    ]
                    print(worst_passing)
                    worst_passing.to_csv(f'worst_passing_configs_{suffix}.csv', index=False)
            
        # Analysis by dimension characteristics
        print("\nPerformance by input dimension:")
        dim_analysis = df.groupby('d_in')['passed'].agg(['count', 'mean'])
        print(dim_analysis.sort_values('mean', ascending=False))
        
        print("\nPerformance by rank:")
        rank_analysis = df.groupby('rank')['passed'].agg(['count', 'mean'])
        print(rank_analysis.sort_values('mean', ascending=False))

def main():
    benchmark = BTTPrecisionBenchmark(rtol=1e-5, atol=1e-8)
    results = benchmark.benchmark_precision()
    benchmark.summarize_results(results)

if __name__ == "__main__":
    main()