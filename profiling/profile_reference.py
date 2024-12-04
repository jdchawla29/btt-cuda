import sys
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Add src/btt to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from btt import BTTLayer, BTTFunction

def btt_layer():
    batch_size = 32
    d_in = 1024  # 32 x 32
    d_out = 4096  # 64 x 64
    tt_rank = 16

    layer = BTTLayer(d_in, d_out, tt_rank)
    x = torch.randn(batch_size, d_in)

    # Move to GPU
    layer = layer.to("cuda")
    x = x.to("cuda")

    y = layer(x)

with profile(
    activities=[
        ProfilerActivity.CPU, 
        ProfilerActivity.CUDA
        ], 
    record_shapes=True,
    ) as prof:
    btt_layer()

print(prof.key_averages().table())