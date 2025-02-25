# BTT-CUDA: Block Tensor-Train API Documentation

## Overview

BTT (Block Tensor-Train) is a memory-efficient parameterization for neural network layers that significantly reduces the parameter count compared to dense layers while maintaining similar performance.

The BTT-CUDA library provides both a PyTorch reference implementation and an optimized CUDA implementation of BTT layers, which can be used as drop-in replacements for standard linear layers in deep learning models.

## Core Concept: What is BTT?

Block Tensor-Train is a specialized matrix factorization technique that decomposes a large weight matrix into multiple smaller matrices, resulting in fewer parameters. Unlike standard Tensor-Train decomposition, BTT uses a block structure that is particularly effective for neural network layers.

The key idea is to represent an input-to-output transformation (weight matrix) as a composition of smaller transformations, reducing the parameter count from O(d_in × d_out) to approximately O(tt_rank × (d_in + d_out)).

### How BTT Works

1. The input dimension d_in is factorized into (m1, m2)
2. The output dimension d_out is factorized into (n1, n2)
3. The BTT decomposition represents the full weight matrix using two smaller matrices:
   - W1 of shape (m2, m1×r1, n1×r2)
   - W2 of shape (n1, m2×r2, n2×r3)

Where r1, r2, and r3 are rank parameters, typically with r1=r3=1, and r2=tt_rank.

The forward computation involves reshaping the input, applying the two matrices with batched matrix multiplications, and reshaping the output.

## API Reference

### Reference Implementation

#### `btt.reference.BTTLayer`

A PyTorch module implementing the Block Tensor-Train layer.

```python
BTTLayer(d_in: int, d_out: int, tt_rank: int, normalize: bool = False)
```

**Parameters:**
- `d_in` (int): Input dimension
- `d_out` (int): Output dimension
- `tt_rank` (int): Tensor-Train rank (higher values increase expressivity but also parameter count)
- `normalize` (bool, optional): Whether to normalize weights during forward pass. Default: False

**Attributes:**
- `W1` (nn.Parameter): First weight matrix of shape (m2, m1*r1, n1*r2)
- `W2` (nn.Parameter): Second weight matrix of shape (n1, m2*r2, n2*r3)
- `shapes` (tuple): Contains (ranks, input_dims, output_dims)

**Method: `forward(x: torch.Tensor) -> torch.Tensor`**

**Input Shape:**
- `x`: (batch_size, d_in)

**Output Shape:**
- (batch_size, d_out)

**Example:**
```python
import torch
from btt.reference import BTTLayer

# Create a BTT layer with 1024 input features, 512 output features, and rank 16
layer = BTTLayer(1024, 512, tt_rank=16)

# Forward pass with a batch of 32 examples
x = torch.randn(32, 1024)
out = layer(x)  # out.shape = (32, 512)
```

### CUDA Implementation

#### `btt_cuda.BTTLayer`

A PyTorch module implementing the Block Tensor-Train layer with CUDA acceleration.

```python
BTTLayer(d_in: int, d_out: int, tt_rank: int, normalize: bool = False)
```

**Parameters:**
- `d_in` (int): Input dimension
- `d_out` (int): Output dimension
- `tt_rank` (int): Tensor-Train rank (higher values increase expressivity but also parameter count)
- `normalize` (bool, optional): Whether to normalize weights during forward pass. Default: False

**Attributes:**
- `W1` (nn.Parameter): First weight matrix of shape (m2, m1*r1, n1*r2)
- `W2` (nn.Parameter): Second weight matrix of shape (n1, m2*r2, n2*r3)
- `shapes` (tuple): Contains (ranks, input_dims, output_dims)

**Method: `forward(x: torch.Tensor) -> torch.Tensor`**

**Input Shape:**
- `x`: (batch_size, d_in)

**Output Shape:**
- (batch_size, d_out)

**Example:**
```python
import torch
from btt_cuda import BTTLayer

# Create a BTT layer with 1024 input features, 512 output features, and rank 16
layer = BTTLayer(1024, 512, tt_rank=16).cuda()

# Forward pass with a batch of 32 examples
x = torch.randn(32, 1024).cuda()
out = layer(x)  # out.shape = (32, 512)
```

## Common Usage Patterns

### 1. As a Drop-in Replacement for Linear Layers

BTT layers can replace standard linear layers to reduce model size:

```python
# Original model with dense layers
class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        # ...
        
# BTT model with reduced parameters
class BTTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = BTTLayer(1024, 512, tt_rank=8)
        self.linear2 = BTTLayer(512, 256, tt_rank=8)
        # ...
```

### 2. In Deep Kernel Learning Models

```python
class BTTFeatureExtractor(nn.Sequential):
    def __init__(self, input_dim: int, tt_rank: int = 8):
        super().__init__()
        self.add_module('btt1', BTTLayer(input_dim, 1000, tt_rank))
        self.add_module('relu1', nn.ReLU())
        self.add_module('btt2', BTTLayer(1000, 500, tt_rank))
        self.add_module('relu2', nn.ReLU())
        self.add_module('btt3', BTTLayer(500, 50, tt_rank))
        self.add_module('relu3', nn.ReLU())
        self.add_module('btt4', BTTLayer(50, 2, tt_rank))
```

### 3. Loading and Saving Models

BTT layers are standard PyTorch modules and support all PyTorch serialization methods:

```python
# Save model with BTT layers
torch.save(model.state_dict(), "btt_model.pt")

# Load model
model = YourBTTModel()
model.load_state_dict(torch.load("btt_model.pt"))
```

## Performance Considerations

### Parameter Efficiency

For a layer with input dimension d_in and output dimension d_out:
- Dense layer: d_in × d_out parameters
- BTT layer with rank r: approximately r × (d_in + d_out) parameters

### Computational Efficiency

1. **CPU Implementation**: 
   - Suitable for testing and development
   - Slower than dense layers for small dimensions
   - Better memory efficiency for large dimensions

2. **CUDA Implementation**:
   - Significant speedup over the CPU implementation (up to 3x faster)
   - Leverages cuBLAS for optimized batch matrix multiplications
   - Efficient for both training and inference

### Choosing the Rank

The `tt_rank` parameter controls the trade-off between model capacity and size:
- Lower ranks (2-4): Maximum parameter reduction, potentially reduced expressivity
- Medium ranks (8-16): Good balance between size and performance
- Higher ranks (32+): Closer to dense layer performance, less parameter savings

### Numerical Precision

The CUDA implementation maintains high numerical precision compared to the reference implementation, with differences typically in the range of 1e-7 to 1e-8.

## Examples

### Basic Usage

```python
import torch
from btt_cuda import BTTLayer

# Create a BTT layer
layer = BTTLayer(1024, 512, tt_rank=16).cuda()

# Forward pass
x = torch.randn(32, 1024).cuda()
output = layer(x)
```

### Integration in a Neural Network

```python
import torch
import torch.nn as nn
from btt_cuda import BTTLayer

class BTTNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tt_rank=8):
        super().__init__()
        self.btt1 = BTTLayer(input_dim, hidden_dim, tt_rank)
        self.activation = nn.ReLU()
        self.btt2 = BTTLayer(hidden_dim, output_dim, tt_rank)
        
    def forward(self, x):
        x = self.btt1(x)
        x = self.activation(x)
        x = self.btt2(x)
        return x
```