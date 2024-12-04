import torch
import torch.nn as nn
from typing import Tuple
import math

class BTTFunction(torch.autograd.Function):
    """Block Tensor-Train matrix multiplication, based on the implementation in mm/btt_mvm.py"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, shapes: Tuple[Tuple[int, ...], ...]) -> torch.Tensor:
        """
        Forward pass based on mm/btt_mvm.py's BlockTeTr implementation.

        Args:
            x: Input tensor of shape (batch_size, prod(ms))
            W1: First weight matrix
            W2: Second weight matrix
            shapes: Tuple of (ranks, input_dims, output_dims)
                ranks: (r1, r2, r3)
                input_dims: (m1, m2)
                output_dims: (n1, n2)
        """
        rs, ms, ns = shapes
        batch_n = x.shape[0]
        out1 = torch.empty(batch_n, ms[1], ns[0] * rs[1], device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.empty(batch_n, ns[0], ns[1] * rs[2], device=x.device, dtype=x.dtype).transpose(0, 1)

        # First transformation
        y = x.reshape(batch_n, ms[1], ms[0], -1)
        y = y.transpose(0, 1)
        y = y.reshape(ms[1], batch_n, ms[0] * rs[0])
        torch.bmm(y, W1, out=out1)

        # Reshape and second transformation
        out1 = out1.reshape(ms[1], batch_n, ns[0], rs[1])
        out1 = out1.transpose(0, 2).contiguous()
        out1 = out1.reshape(ns[0], batch_n, ms[1] * rs[1])
        torch.bmm(out1, W2, out=out2)

        # Final reshape
        out2 = out2.reshape(ns[0], batch_n, ns[1], rs[2])
        out2 = out2.transpose(0, 1)
        out2 = out2.reshape(batch_n, -1)

        ctx.save_for_backward(x, W1, W2, out1)
        ctx.shapes = shapes
        return out2

    @staticmethod
    def backward(ctx, grad):
        x, W1, W2, out1 = ctx.saved_tensors
        rs, ms, ns = ctx.shapes
        B = x.shape[0]
        grad_re = grad.reshape(B, ns[0], ns[1]).transpose(1, 0)
        aux = torch.empty(B, ns[0], rs[1] * ms[1], device=x.device, dtype=x.dtype).transpose(1, 0)
        dx = torch.empty(B, ms[1], ms[0], device=x.device, dtype=x.dtype).transpose(1, 0)

        # Compute gradients
        torch.bmm(grad_re, W2.transpose(1, 2), out=aux)
        aux = aux.reshape(ns[0], B, ms[1], rs[1]).transpose(0, 2).contiguous()
        aux = aux.reshape(ms[1], B, ns[0] * rs[1])
        torch.bmm(aux, W1.transpose(1, 2), out=dx)
        dx = dx.reshape(ms[1], B, ms[0] * rs[0])
        dx = dx.transpose(1, 0).reshape(B, -1)

        # Compute weight gradients
        x_res = x.reshape(B, ms[1], ms[0]).transpose(0, 1)
        dW1 = torch.bmm(aux.transpose(1, 2), x_res).transpose(1, 2)
        dW2 = torch.bmm(grad_re.transpose(1, 2), out1).transpose(1, 2)

        return dx, dW1, dW2, None

class BTTLayer(nn.Module):
    """Block Tensor-Train Layer implementation based on ops/operators.py"""
    def __init__(self, d_in: int, d_out: int, tt_rank: int, normalize: bool = False):
        super().__init__()

        # Factor dimensions using sqrt for now
        self.m1 = int(math.sqrt(d_in))
        self.m2 = d_in // self.m1
        self.n1 = int(math.sqrt(d_out))
        self.n2 = int(math.sqrt(d_out))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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