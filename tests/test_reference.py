import sys
import os
import pytest
import torch
import torch.nn as nn

# Add src/btt to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from btt import BTTLayer, BTTFunction

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dtype():
    return torch.float64 # Use double precision for testing

def test_shapes():
    """Test that BTT layer produces correct output shapes."""
    batch_size = 32
    d_in = 256  # 16 x 16
    d_out = 1024  # 32 x 32
    tt_rank = 8
    
    layer = BTTLayer(d_in, d_out, tt_rank)
    x = torch.randn(batch_size, d_in)
    y = layer(x)
    
    assert y.shape == (batch_size, d_out)
    
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

def test_gradient_numerically(device, dtype):
    """Test gradients using torch.autograd.gradcheck."""
    d_in = 16  # 4 x 4 
    d_out = 36  # 6 x 6
    tt_rank = 2
    
    layer = BTTLayer(d_in, d_out, tt_rank).to(device=device, dtype=dtype)
    
    def func(x, W1, W2):
        shapes = layer.shapes
        return BTTFunction.apply(x, W1, W2, shapes)
    
    x = torch.randn(2, d_in, device=device, dtype=dtype, requires_grad=True)
    W1 = layer.W1.detach().requires_grad_()
    W2 = layer.W2.detach().requires_grad_()

    assert torch.autograd.gradcheck(func, (x, W1, W2), eps=1e-6, atol=1e-4)

def test_normalization(device, dtype):
    """Test that normalization keeps outputs stable."""
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

def test_parameter_count():
    """Test that BTT layer uses fewer parameters than dense layer."""
    d_in = 1024  # 32 x 32
    d_out = 1024  # 32 x 32
    tt_rank = 4
    
    dense = nn.Linear(d_in, d_out, bias=False)
    btt = BTTLayer(d_in, d_out, tt_rank)
    
    dense_params = sum(p.numel() for p in dense.parameters())
    btt_params = sum(p.numel() for p in btt.parameters())
    
    # BTT should use significantly fewer parameters
    assert btt_params < dense_params * 0.5, f"BTT params ({btt_params}) should be much less than dense ({dense_params})"

def test_numerical_stability():
    """Test numerical stability with different input scales."""
    d_in = 256
    d_out = 256
    tt_rank = 8
    batch_size = 32
    
    layer = BTTLayer(d_in, d_out, tt_rank, normalize=True)
    
    # Test with different input scales
    scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    outputs = []
    
    for scale in scales:
        x = torch.randn(batch_size, d_in) * scale
        y = layer(x)
        norm = torch.norm(y) / torch.norm(x)
        outputs.append(norm.item())
    
    # Check that output/input ratio remains relatively stable across scales
    outputs = torch.tensor(outputs)
    relative_variation = (outputs.max() - outputs.min()) / outputs.mean()
    assert relative_variation < 1.0, f"Output/input ratio varies too much: {relative_variation}"