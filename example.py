import torch
from btt.reference import BTTLayer as CPULayer
from btt_cuda import BTTLayer as GPULayer

# Test CPU implementation
cpu_layer = CPULayer(64, 64, tt_rank=4)
x_cpu = torch.randn(32, 64)
out_cpu = cpu_layer(x_cpu)
print('CPU output shape:', out_cpu.shape)

# Test CUDA implementation
gpu_layer = GPULayer(64, 64, tt_rank=4).cuda()
x_gpu = torch.randn(32, 64).cuda()
out_gpu = gpu_layer(x_gpu)
print('GPU output shape:', out_gpu.shape)