# BTT-CUDA: Block Tensor-Train for PyTorch with CUDA Acceleration

BTT-CUDA provides memory-efficient parameterization for neural network layers through Block Tensor-Train decomposition, with highly optimized CUDA implementations.

## Features

- **Memory Efficient**: Reduces parameter count from O(d_in × d_out) to approximately O(tt_rank × (d_in + d_out))
- **Drop-in Replacement**: Use BTT layers as direct replacements for standard linear layers
- **CUDA Accelerated**: Optimized CUDA implementation using cuBLAS for fast forward and backward passes
- **Numerically Stable**: High precision maintained between reference and CUDA implementations
- **PyTorch Native**: Full compatibility with PyTorch's autograd system

## Installation

```bash
pip install .
```

## Quick Start

```python
import torch
from btt_cuda import BTTLayer

# Create a BTT layer with 1024 input features, 512 output features, and rank 16
layer = BTTLayer(1024, 512, tt_rank=16).cuda()

# Forward pass with a batch of 32 examples
x = torch.randn(32, 1024).cuda()
out = layer(x)  # out.shape = (32, 512)
```

## Comparing CPU and CUDA Implementations

```python
import torch
from btt.reference import BTTLayer as PyBTTLayer
from btt_cuda import BTTLayer as CUDABTTLayer

# Create a BTT layer with CPU implementation
py_layer = PyBTTLayer(1024, 512, tt_rank=16)

# Create a BTT layer with CUDA implementation
cuda_layer = CUDABTTLayer(1024, 512, tt_rank=16).cuda()

# Copy weights from CPU to CUDA implementation for fair comparison
cuda_layer.W1.data = py_layer.W1.data.cuda()
cuda_layer.W2.data = py_layer.W2.data.cuda()

# Input tensor
x = torch.randn(32, 1024)
x_cuda = x.cuda()

# Forward pass on CPU
py_out = py_layer(x)

# Forward pass on CUDA
cuda_out = cuda_layer(x_cuda).cpu()

# Verify outputs match
print(f"Max difference: {(py_out - cuda_out).abs().max().item()}")
```

## Documentation

See [API Documentation](docs/api.md) for detailed usage instructions and examples.

## Benchmarks

The CUDA implementation provides significant speedups compared to the reference implementation:

- Forward pass: Up to 3x faster
- Backward pass: Up to 2.5x faster
- Memory usage: Significantly reduced for large models

## Citation

If you use this library in your research, please cite:

```
@misc{BTT-CUDA,
  author = {Jaideep Chawla},
  title = {BTT-CUDA: Block Tensor-Train for PyTorch with CUDA Acceleration},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/jaideepchawla/btt-cuda}},
}
```

## License

MIT License
