from ._cuda import forward, backward
from .btt_cuda_layer import BTTLayer, BTTFunction

try:
    from ._cuda_optimized import forward as forward_optimized
    from ._cuda_optimized import backward as backward_optimized
    from .btt_cuda_optimized_layer import BTTLayerOptimized, BTTFunctionOptimized
    __all__ = ['BTTLayer', 'BTTFunction', 'BTTLayerOptimized', 'BTTFunctionOptimized', 'forward', 'backward', 'forward_optimized', 'backward_optimized']
except ImportError:
    __all__ = ['BTTLayer', 'BTTFunction', 'forward', 'backward']