#include <torch/extension.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

static cublasHandle_t cublas_handle;
static bool cublas_initialized = false;

inline void checkCublasStatus(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS error at: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void init_cublas() {
    if (!cublas_initialized) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        checkCublasStatus(status, "cublasCreate");
        cublas_initialized = true;
    }
}

// GEMM wrapper: C = A*B for row-major data using cublas (which is column-major).
// Dimensions: A(M,K), B(K,N), C(M,N)
static void gemm_row_major_float(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    // We pass A^T and B^T to cublas because it expects column-major.
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        B, K,
        A, M,
        &beta,
        C, N
    );
    checkCublasStatus(status, "cublasSgemm in gemm_row_major_float");
}

// Forward pass: also return out1 for backward
std::pair<torch::Tensor, torch::Tensor> btt_cuda_forward(
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    int m1, int m2,
    int n1, int n2,
    int r1, int r2
) {
    init_cublas();
    int B = input.size(0);

    // x_3d = (B,m2,m1)
    auto x_3d = input.reshape({B,m2,m1});

    // Compute out1 = x_3d * W1
    // W1: (m2,m1,n1*r2)
    // For each slice along m2:
    // out1_bim: (B,m2,m1) * (m1,n1*r2) = (B,m2,n1*r2)
    // We'll loop over m2 to avoid complicated indexing:

    auto out1 = torch::empty({B,m2,n1*r2}, input.options());

    for (int i=0; i<m2; i++){
        auto A = x_3d.select(1,i).contiguous(); // (B,m1)
        auto W1_i = W1.select(0,i).contiguous().reshape({m1,n1*r2}); // (m1,n1*r2)
        auto C = torch::empty({B,n1*r2}, input.options());
        gemm_row_major_float(
            B, n1*r2, m1,
            A.data_ptr<float>(), m1,
            W1_i.data_ptr<float>(), n1*r2,
            C.data_ptr<float>(), n1*r2
        );
        out1.select(1,i).copy_(C);
    }

    // Now out1: (B,m2,n1*r2)
    // out1_3d = (B,n1,m2*r2)
    auto out1_3d = out1.transpose(1,2).contiguous().reshape({B,n1,m2*r2});

    // Compute out2 = out1_3d * W2
    // W2: (n1,m2*r2,n2)
    auto out2 = torch::empty({B,n1,n2}, input.options());
    for (int i=0; i<n1; i++){
        auto A = out1_3d.select(1,i).contiguous(); // (B,m2*r2)
        auto W2_i = W2.select(0,i).contiguous().reshape({m2*r2,n2}); // (m2*r2,n2)
        auto C = torch::empty({B,n2}, input.options());
        gemm_row_major_float(
            B, n2, m2*r2,
            A.data_ptr<float>(), m2*r2,
            W2_i.data_ptr<float>(), n2,
            C.data_ptr<float>(), n2
        );
        out2.select(1,i).copy_(C);
    }

    auto output = out2.reshape({B, n1*n2});

    return {output, out1};
}

// Backward pass
std::vector<torch::Tensor> btt_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    torch::Tensor out1,
    int m1, int m2,
    int n1, int n2,
    int r1, int r2
) {
    init_cublas();
    int B = input.size(0);

    // grad_output: (B,n1*n2) -> (B,n1,n2)
    auto grad_out2 = grad_output.reshape({B,n1,n2});
    auto out1_3d = out1.transpose(1,2).contiguous().reshape({B,n1,m2*r2});

    // grad_W2 = einsum('bik,bij->ikj', out1_3d, grad_out2)
    // Loop over i:
    auto grad_W2 = torch::zeros({n1,m2*r2,n2}, input.options());
    for (int i=0; i<n1; i++){
        // out1_3d_i: (B,m2*r2)
        auto out1_3d_i = out1_3d.select(1,i).contiguous();
        // grad_out2_i: (B,n2)
        auto grad_out2_i = grad_out2.select(1,i).contiguous();
        // (m2*r2,B)*(B,n2) = (m2*r2,n2) after transpose logic
        // We do (B,m2*r2)^T * (B,n2) in row-major:
        // Just call gemm with A=grad_out2_i and B=out1_3d_i but swapped?
        // We want: grad_W2[i,:,:] = out1_3d_i^T * grad_out2_i
        // out1_3d_i: (B,m2*r2), grad_out2_i:(B,n2)
        // gemm: (m2*r2) = K, B = M dimension
        // We'll do: (m2*r2,B)*(B,n2) by transposing arguments:
        // Let's rearrange so A=(B,m2*r2), B=(B,n2)
        // We want C=(m2*r2,n2)
        // If we pass A=grad_out2_i(B,n2) and B=out1_3d_i(B,m2*r2),
        // C = B^T * A = (m2*r2,n2) if we treat carefully:
        // Use gemm_row_major_float:
        // gemm_row_major_float(M=m2*r2, N=n2, K=B, A=out1_3d_i(B,m2*r2), B=grad_out2_i(B,n2))
        // Wait M corresponds to out1_3d_i dimension (m2*r2), N=n2, K=B
        // A: (B,m2*r2), B: (B,n2) → we need K dimension consistent = B
        // Let's transpose logic: We want C(m2*r2,n2):
        // Consider: C = out1_3d_i^T * grad_out2_i
        // out1_3d_i^T is (m2*r2,B)
        // grad_out2_i is (B,n2)
        // So M=m2*r2, N=n2, K=B
        // A=(B,m2*r2), B=(B,n2) but we must swap them to get proper orientation.

        auto W2_part = torch::empty({m2*r2,n2}, input.options());

        // gemm_row_major_float(M,N,K,A,lda,B,ldb,C,ldc)
        // M=m2*r2, N=n2, K=B
        // A should be (M,K)=(m2*r2,B), we have out1_3d_i=(B,m2*r2), so we transpose it by passing as B and A swapped:
        // Pass out1_3d_i as B, grad_out2_i as A with roles swapped in gemm call:
        // We'll first transpose dimensions by logic in gemm:
        gemm_row_major_float(
            m2*r2, n2, B,
            out1_3d_i.data_ptr<float>(), m2*r2, // A(M,K) interpreted
            grad_out2_i.data_ptr<float>(), n2,
            W2_part.data_ptr<float>(), n2
        );

        grad_W2.select(0,i).copy_(W2_part);
    }

    // grad_out1_3d = einsum('bij,ikj->bik', grad_out2, W2)
    // Loop over i for W2:
    auto grad_out1_3d = torch::empty({B,n1,m2*r2}, input.options());
    for (int i=0; i<n1; i++){
        auto grad_out2_i = grad_out2.select(1,i).contiguous(); // (B,n2)
        auto W2_i = W2.select(0,i).contiguous().reshape({m2*r2,n2}); // (m2*r2,n2)
        // We want: grad_out1_3d_i(b,k) = sum_j grad_out2_i(b,j)*W2_i(i,k,j)
        // C = grad_out2_i * W2_i^T:
        // (B,n2)*(n2,m2*r2)=(B,m2*r2)
        auto C = torch::empty({B,m2*r2}, input.options());

        gemm_row_major_float(
            B, m2*r2, n2,
            grad_out2_i.data_ptr<float>(), n2,
            W2_i.data_ptr<float>(), n2, // W2_i (m2*r2,n2) -> transposed in gemm
            C.data_ptr<float>(), m2*r2
        );
        grad_out1_3d.select(1,i).copy_(C);
    }

    // grad_out1 = reshape to (B,m2,n1*r2)
    auto grad_out1 = grad_out1_3d.reshape({B,n1,m2,r2}).transpose(1,2).reshape({B,m2,n1*r2});

    // grad_W1 = einsum('bim,bij->imj', x_3d, grad_out1)
    // Loop over m2:
    auto x_3d = input.reshape({B,m2,m1});
    auto grad_W1 = torch::empty({m2,m1,n1*r2}, input.options());
    for (int i=0; i<m2; i++){
        auto x_slice = x_3d.select(1,i).contiguous();     // (B,m1)
        auto g_slice = grad_out1.select(1,i).contiguous(); // (B,n1*r2)

        auto W1_part = torch::empty({m1,n1*r2}, input.options());
        // sum over B: (m1,B)*(B,n1*r2) = (m1,n1*r2)
        // gemm_row_major_float does C= A*B
        // If A=(B,m1), B=(B,n1*r2), we need A^T * B:
        // We'll do: M=m1,N=n1*r2,K=B
        // A=x_slice(B,m1), B=g_slice(B,n1*r2)
        // We want x_slice^T*g_slice: (m1,B)*(B,n1*r2)
        gemm_row_major_float(
            m1, n1*r2, B,
            x_slice.data_ptr<float>(), m1,
            g_slice.data_ptr<float>(), n1*r2,
            W1_part.data_ptr<float>(), n1*r2
        );

        grad_W1.select(0,i).copy_(W1_part);
    }

    // grad_x = einsum('bij,imj->bim', grad_out1, W1)
    // Similar to above:
    auto grad_x = torch::empty({B,m2,m1}, input.options());
    for (int i=0; i<m2; i++){
        auto g_slice = grad_out1.select(1,i).contiguous(); // (B,n1*r2)
        auto W1_part = W1.select(0,i).contiguous().reshape({m1,n1*r2}); // (m1,n1*r2)

        // grad_x_slice(b,m1) = g_slice(b,n1*r2)*W1_part^T(n1*r2,m1)
        // (B,n1*r2)*(n1*r2,m1)=(B,m1)
        auto C = torch::empty({B,m1}, input.options());
        gemm_row_major_float(
            B, m1, n1*r2,
            g_slice.data_ptr<float>(), n1*r2,
            W1_part.data_ptr<float>(), n1*r2,
            C.data_ptr<float>(), m1
        );
        grad_x.select(1,i).copy_(C);
    }

    auto grad_input = grad_x.reshape({B,m1*m2});

    return {grad_input, grad_W1, grad_W2};
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "BTT CUDA with cuBLAS optimized";
    
    m.def("forward", &btt_cuda_forward, "BTT forward (CUDA cuBLAS)",
          py::arg("input"), py::arg("W1"), py::arg("W2"),
          py::arg("m1"), py::arg("m2"),
          py::arg("n1"), py::arg("n2"),
          py::arg("r1"), py::arg("r2")
    );
    
    m.def("backward", &btt_cuda_backward, "BTT backward (CUDA cuBLAS)",
          py::arg("grad_output"), py::arg("input"),
          py::arg("W1"), py::arg("W2"), py::arg("out1"),
          py::arg("m1"), py::arg("m2"),
          py::arg("n1"), py::arg("n2"),
          py::arg("r1"), py::arg("r2")
    );
}
