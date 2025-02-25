# setup.py
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get absolute paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
cuda_sources = [
    os.path.join(curr_dir, 'src/btt_cuda/cuda/btt_cuda.cpp'),
    # os.path.join(curr_dir, 'src/btt_cuda/cuda/btt_cuda_kernel.cu')
]

cuda_optimized_sources = [
    os.path.join(curr_dir, 'src/btt_cuda/cuda/btt_cuda_optimized.cpp'),
]

setup(
    name='btt',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    ext_modules=[
        CUDAExtension(
            name='btt_cuda._cuda',
            sources=cuda_sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        ),
        CUDAExtension(
            name='btt_cuda._cuda_optimized',
            sources=cuda_optimized_sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-use_fast_math', '-Xptxas=-v']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)