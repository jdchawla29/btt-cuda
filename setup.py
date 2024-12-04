# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

print(f"Current directory: {os.getcwd()}")
cuda_ext = CUDAExtension(
    name='btt.cuda_impl',
    sources=[
        'src/cuda/btt_cuda.cpp',
        'src/cuda/btt_cuda_kernel.cu',
    ],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': ['-O3']
    },
    verbose=True  # Add verbose flag
)
print(f"Extension sources: {cuda_ext.sources}")
print(f"Extension name: {cuda_ext.name}")

setup(
    name='btt',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[cuda_ext],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)