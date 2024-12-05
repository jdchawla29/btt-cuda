#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor btt_cuda_forward(
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    const int m1, const int m2,
    const int n1, const int n2,
    const int r1, const int r2);

std::vector<torch::Tensor> btt_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    torch::Tensor intermediate,
    const int m1, const int m2,
    const int n1, const int n2,
    const int r1, const int r2);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Block Tensor-Train CUDA implementation";  // Optional module docstring
    
    m.def("forward", &btt_cuda_forward, "BTT forward (CUDA)",
          py::arg("input"),
          py::arg("W1"),
          py::arg("W2"),
          py::arg("m1"),
          py::arg("m2"),
          py::arg("n1"),
          py::arg("n2"),
          py::arg("r1"),
          py::arg("r2")
    );
    
    m.def("backward", &btt_cuda_backward, "BTT backward (CUDA)",
          py::arg("grad_output"),
          py::arg("input"),
          py::arg("W1"),
          py::arg("W2"),
          py::arg("intermediate"),
          py::arg("m1"),
          py::arg("m2"),
          py::arg("n1"),
          py::arg("n2"),
          py::arg("r1"),
          py::arg("r2")
    );
}