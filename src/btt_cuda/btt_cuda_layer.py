import math
import torch
import torch.nn as nn
from . import forward, backward

class BTTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2, shapes):
        rs, ms, ns = shapes
        m1, m2 = ms
        n1, n2 = ns
        r1, r2, r3 = rs
        
        # Call CUDA (cuBLAS) forward
        out, out1 = forward(x, W1, W2, m1, m2, n1, n2, r1, r2)
        
        # Save for backward
        ctx.save_for_backward(x, W1, W2, out1)
        ctx.shapes = shapes
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, out1 = ctx.saved_tensors
        rs, ms, ns = ctx.shapes
        m1, m2 = ms
        n1, n2 = ns
        r1, r2, r3 = rs

        grad_input, grad_W1, grad_W2 = backward(grad_output, x, W1, W2, out1, m1, m2, n1, n2, r1, r2)
        return grad_input, grad_W1, grad_W2, None
    
def closest_factors(n):
    root = int(math.isqrt(n))
    # Start from the integer sqrt and go downwards.
    for i in range(root, 0, -1):
        if n % i == 0:
            return n // i, i
    # Fallback (should never actually reach here since 1 is always a divisor)
    return n, 1

class BTTLayer(nn.Module):
    """Block Tensor-Train Layer with CUDA implementation"""
    def __init__(self, d_in: int, d_out: int, tt_rank: int, normalize: bool = False):
        super().__init__()

        # Find factors closest to sqrt
        self.m1, self.m2 = closest_factors(d_in)
        self.n1, self.n2 = closest_factors(d_out)

        # Verify dimensions
        assert self.m1 * self.m2 == d_in, f"Input dim {d_in} must be factorizable"
        assert self.n1 * self.n2 == d_out, f"Output dim {d_out} must be factorizable"

        # Initialize shapes
        self.rs = (1, tt_rank, 1)  # ranks
        self.ms = (self.m1, self.m2)  # input dims
        self.ns = (self.n1, self.n2)  # output dims
        self.shapes = (self.rs, self.ms, self.ns)

        # Initialize weights
        # W1: (m2, m1*r1, n1*r2)  
        # W2: (n1, m2*r2, n2*r3)
        self.W1 = nn.Parameter(torch.randn(self.m2, self.m1 * self.rs[0], self.n1 * self.rs[1]))
        self.W2 = nn.Parameter(torch.randn(self.n1, self.m2 * self.rs[1], self.n2 * self.rs[2]))

        # Initialize with proper scaling
        d_in_w1 = self.m1 * self.rs[0]
        d_in_w2 = self.m2 * self.rs[1]
        nn.init.xavier_normal_(self.W1, gain=1.0 / math.sqrt(d_in_w1))
        nn.init.xavier_normal_(self.W2, gain=1.0 / math.sqrt(d_in_w2))

        self.normalize = normalize
        self.tt_rank = tt_rank
        self.max_rms = (min(d_in, d_out) * d_out / (d_out * d_in * d_in))**0.5

    def forward(self, x):
        if self.normalize:
            # Compute RMS values
            rms_W0 = torch.sqrt(torch.mean(self.W1 ** 2.) + 1e-8)
            rms_W1 = torch.sqrt(torch.mean(self.W2 ** 2.) + 1e-8)

            # Scale matrices
            W1 = self.W1 / max(1, rms_W0 / self.max_rms)
            W2 = self.W2 / max(1, rms_W1 / self.max_rms)
        else:
            W1, W2 = self.W1, self.W2

        return BTTFunction.apply(x, W1, W2, self.shapes)

    def extra_repr(self) -> str:
        return f'in={self.m1}×{self.m2}, out={self.n1}×{self.n2}, rank={self.tt_rank}'