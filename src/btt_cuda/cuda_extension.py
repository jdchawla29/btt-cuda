import os
import torch
from torch.utils.cpp_extension import load

# Get absolute path to the current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to source files
btt_cuda_path = os.path.join(curr_dir, 'cuda/btt_cuda_inline.cpp')
btt_cuda_optimized_path = os.path.join(curr_dir, 'cuda/btt_cuda_optimized_inline.cpp')

# Load the basic CUDA extension
_cuda = load(
    name='btt_cuda._cuda_inline',
    sources=[btt_cuda_path],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    verbose=True,
    with_cuda=True,
    build_directory=os.path.join(curr_dir, 'build')
)

# Load the optimized CUDA extension
_cuda_optimized = load(
    name='btt_cuda._cuda_optimized_inline',
    sources=[btt_cuda_optimized_path],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-use_fast_math', '-Xptxas=-v'],
    verbose=True,
    with_cuda=True,
    build_directory=os.path.join(curr_dir, 'build')
)

# Export the functions with the original names for compatibility
forward = _cuda.btt_cuda_forward
backward = _cuda.btt_cuda_backward
forward_optimized = _cuda_optimized.btt_cuda_forward_optimized
backward_optimized = _cuda_optimized.btt_cuda_backward_optimized