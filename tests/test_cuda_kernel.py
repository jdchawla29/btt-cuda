# test_btt_cuda.py
import sys
import os
import pytest
import torch
import torch.nn as nn

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from btt_cuda import BTTLayer, BTTFunction
from btt import BTTLayer as CPULayer  

@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")

@pytest.fixture(scope="module")
def dtype():
    return torch.float64  # Use double precision for testing

def test_shapes(device):
    """Test that CUDA BTT layer produces correct output shapes."""
    batch_size = 32
    d_in = 256  # 16 x 16
    d_out = 1024  # 32 x 32
    tt_rank = 8
    
    layer = BTTLayer(d_in, d_out, tt_rank).to(device)
    x = torch.randn(batch_size, d_in, device=device)
    y = layer(x)
    
    assert y.shape == (batch_size, d_out)

def test_cuda_vs_cpu(device, dtype):
    """Test that CUDA implementation matches CPU implementation."""
    batch_size = 4
    d_in = 64  # 8 x 8
    d_out = 144  # 12 x 12
    tt_rank = 4
    
    # Create both CPU and CUDA layers with same initialization
    torch.manual_seed(42)
    cpu_layer = CPULayer(d_in, d_out, tt_rank).to(dtype=dtype)
    
    torch.manual_seed(42)
    cuda_layer = BTTLayer(d_in, d_out, tt_rank).to(device=device, dtype=dtype)
    
    # Test with same input
    x = torch.randn(batch_size, d_in, dtype=dtype)
    y_cpu = cpu_layer(x)
    y_cuda = cuda_layer(x.to(device))
    
    # Check outputs match
    assert torch.allclose(y_cpu, y_cuda.cpu(), rtol=1e-5, atol=1e-5)

def test_backward(device, dtype):
    """Test that backward pass works and gradients are computed correctly."""
    batch_size = 4
    d_in = 64  # 8 x 8
    d_out = 144  # 12 x 12
    tt_rank = 4
    
    layer = BTTLayer(d_in, d_out, tt_rank).to(device=device, dtype=dtype)
    x = torch.randn(batch_size, d_in, device=device, dtype=dtype, requires_grad=True)
    
    # Forward pass
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert layer.W1.grad is not None
    assert layer.W2.grad is not None
    
    # Check gradient shapes
    assert x.grad.shape == x.shape
    assert layer.W1.grad.shape == layer.W1.shape
    assert layer.W2.grad.shape == layer.W2.shape

def test_normalization(device, dtype):
    """Test that normalization works on GPU."""
    batch_size = 8
    d_in = 100
    d_out = 100
    tt_rank = 4
    
    layer = BTTLayer(d_in, d_out, tt_rank, normalize=True).to(device=device, dtype=dtype)
    x = torch.randn(batch_size, d_in, device=device, dtype=dtype)
    
    # Do multiple forward passes
    outputs = []
    for _ in range(10):
        y = layer(x)
        outputs.append(y)
        
    # Check that outputs are stable
    outputs = torch.stack(outputs)
    max_diff = (outputs.max(0).values - outputs.min(0).values).max()
    assert max_diff < 1e-5

def test_large_batch(device):
    """Test with large batch sizes to stress CUDA implementation."""
    d_in = 256
    d_out = 256
    tt_rank = 8
    large_batch = 2048
    
    layer = BTTLayer(d_in, d_out, tt_rank).to(device)
    x = torch.randn(large_batch, d_in, device=device)
    
    # Should handle large batch without OOM
    y = layer(x)
    assert y.shape == (large_batch, d_out)

def test_numerical_stability(device):
    """Test numerical stability with different input scales on GPU."""
    d_in = 256
    d_out = 256
    tt_rank = 8
    batch_size = 32
    
    layer = BTTLayer(d_in, d_out, tt_rank, normalize=True).to(device)
    
    # Test with different input scales
    scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    outputs = []
    
    for scale in scales:
        x = torch.randn(batch_size, d_in, device=device) * scale
        y = layer(x)
        norm = torch.norm(y) / torch.norm(x)
        outputs.append(norm.item())
    
    # Check that output/input ratio remains relatively stable
    outputs = torch.tensor(outputs)
    relative_variation = (outputs.max() - outputs.min()) / outputs.mean()
    assert relative_variation < 1.0, f"Output/input ratio varies too much: {relative_variation}"