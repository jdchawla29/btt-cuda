#include <torch/extension.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

static cublasHandle_t cublas_handle;
static bool cublas_initialized = false;
static cudaStream_t stream;
static bool stream_initialized = false;

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
        
        // Set math mode to allow Tensor Cores on newer GPUs
        status = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
        checkCublasStatus(status, "cublasSetMathMode");
        
        cublas_initialized = true;
    }
}

void init_cuda() {
    init_cublas();
    if (!stream_initialized) {
        cudaStreamCreate(&stream);
        cublasSetStream(cublas_handle, stream);
        stream_initialized = true;
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

// Batched GEMM wrapper for the first part of forward pass
static void batched_gemm_for_w1(
    int B, int m2, int m1, int n1, int r2,
    torch::Tensor x_3d,
    torch::Tensor W1,
    torch::Tensor out1
) {
    // Prepare arrays for batch operation
    std::vector<const float*> A_array(m2);
    std::vector<const float*> B_array(m2);
    std::vector<float*> C_array(m2);
    
    // Create temporary holders for device pointers
    auto d_A_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_B_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_C_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Fill host arrays with device pointers
    for (int i = 0; i < m2; i++) {
        A_array[i] = x_3d.select(1, i).contiguous().data_ptr<float>();
        B_array[i] = W1.select(0, i).contiguous().reshape({m1, n1*r2}).data_ptr<float>();
        C_array[i] = out1.select(1, i).data_ptr<float>();
    }
    
    // Copy host arrays to device
    cudaMemcpy(d_A_array.data_ptr<int64_t>(), A_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array.data_ptr<int64_t>(), B_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array.data_ptr<int64_t>(), C_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    
    // Execute batched GEMM
    cublasStatus_t status = cublasSgemmBatched(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        n1*r2, B, m1,
        &alpha,
        (const float**)d_B_array.data_ptr<int64_t>(), m1,
        (const float**)d_A_array.data_ptr<int64_t>(), B,
        &beta,
        (float**)d_C_array.data_ptr<int64_t>(), n1*r2,
        m2
    );
    checkCublasStatus(status, "cublasSgemmBatched for W1");
}

// Batched GEMM wrapper for the second part of forward pass
static void batched_gemm_for_w2(
    int B, int n1, int m2, int r2, int n2,
    torch::Tensor out1_3d,
    torch::Tensor W2,
    torch::Tensor out2
) {
    // Prepare arrays for batch operation
    std::vector<const float*> A_array(n1);
    std::vector<const float*> B_array(n1);
    std::vector<float*> C_array(n1);
    
    // Create temporary holders for device pointers
    auto d_A_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_B_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_C_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Fill host arrays with device pointers
    for (int i = 0; i < n1; i++) {
        A_array[i] = out1_3d.select(1, i).contiguous().data_ptr<float>();
        B_array[i] = W2.select(0, i).contiguous().reshape({m2*r2, n2}).data_ptr<float>();
        C_array[i] = out2.select(1, i).data_ptr<float>();
    }
    
    // Copy host arrays to device
    cudaMemcpy(d_A_array.data_ptr<int64_t>(), A_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array.data_ptr<int64_t>(), B_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array.data_ptr<int64_t>(), C_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    
    // Execute batched GEMM
    cublasStatus_t status = cublasSgemmBatched(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        n2, B, m2*r2,
        &alpha,
        (const float**)d_B_array.data_ptr<int64_t>(), m2*r2,
        (const float**)d_A_array.data_ptr<int64_t>(), B,
        &beta,
        (float**)d_C_array.data_ptr<int64_t>(), n2,
        n1
    );
    checkCublasStatus(status, "cublasSgemmBatched for W2");
}

// Forward pass: also return out1 for backward
std::pair<torch::Tensor, torch::Tensor> btt_cuda_forward_optimized(
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    int m1, int m2,
    int n1, int n2,
    int r1, int r2
) {
    init_cuda();
    int B = input.size(0);
    
    // Ensure inputs are on GPU and contiguous
    input = input.contiguous();
    W1 = W1.contiguous();
    W2 = W2.contiguous();
    
    // x_3d = (B,m2,m1)
    auto x_3d = input.reshape({B, m2, m1});
    
    // Pre-allocate output buffer for batch operation
    auto out1 = torch::empty({B, m2, n1*r2}, input.options());
    
    // Use batched GEMM to compute out1 = x_3d * W1
    batched_gemm_for_w1(B, m2, m1, n1, r2, x_3d, W1, out1);
    
    // Now out1: (B,m2,n1*r2)
    // out1_3d = (B,n1,m2*r2)
    auto out1_3d = out1.transpose(1, 2).contiguous().reshape({B, n1, m2*r2});
    
    // Pre-allocate output buffer for second batch operation
    auto out2 = torch::empty({B, n1, n2}, input.options());
    
    // Use batched GEMM to compute out2 = out1_3d * W2
    batched_gemm_for_w2(B, n1, m2, r2, n2, out1_3d, W2, out2);
    
    // Reshape output
    auto output = out2.reshape({B, n1*n2});
    
    // Ensure all CUDA operations are complete
    cudaStreamSynchronize(stream);
    
    return {output, out1};
}

// Batched GEMM wrapper for grad_W2 computation
static void batched_gemm_for_grad_W2(
    int B, int n1, int m2, int r2, int n2,
    torch::Tensor out1_3d,
    torch::Tensor grad_out2,
    torch::Tensor grad_W2
) {
    // Implementation similar to batched_gemm_for_w1 but for grad_W2 computation
    std::vector<const float*> A_array(n1);
    std::vector<const float*> B_array(n1);
    std::vector<float*> C_array(n1);
    
    auto d_A_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_B_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_C_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    for (int i = 0; i < n1; i++) {
        A_array[i] = out1_3d.select(1, i).contiguous().data_ptr<float>();
        B_array[i] = grad_out2.select(1, i).contiguous().data_ptr<float>();
        C_array[i] = grad_W2.select(0, i).data_ptr<float>();
    }
    
    cudaMemcpy(d_A_array.data_ptr<int64_t>(), A_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array.data_ptr<int64_t>(), B_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array.data_ptr<int64_t>(), C_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    
    cublasStatus_t status = cublasSgemmBatched(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n2, m2*r2, B,
        &alpha,
        (const float**)d_B_array.data_ptr<int64_t>(), n2,
        (const float**)d_A_array.data_ptr<int64_t>(), m2*r2,
        &beta,
        (float**)d_C_array.data_ptr<int64_t>(), n2,
        n1
    );
    checkCublasStatus(status, "cublasSgemmBatched for grad_W2");
}

// Batched GEMM wrapper for grad_out1_3d computation
static void batched_gemm_for_grad_out1_3d(
    int B, int n1, int m2, int r2, int n2,
    torch::Tensor grad_out2,
    torch::Tensor W2,
    torch::Tensor grad_out1_3d
) {
    // Implementation similar to batched_gemm_for_w2 but for grad_out1_3d computation
    std::vector<const float*> A_array(n1);
    std::vector<const float*> B_array(n1);
    std::vector<float*> C_array(n1);
    
    auto d_A_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_B_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_C_array = torch::empty({n1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    for (int i = 0; i < n1; i++) {
        A_array[i] = grad_out2.select(1, i).contiguous().data_ptr<float>();
        B_array[i] = W2.select(0, i).contiguous().reshape({m2*r2, n2}).data_ptr<float>();
        C_array[i] = grad_out1_3d.select(1, i).data_ptr<float>();
    }
    
    cudaMemcpy(d_A_array.data_ptr<int64_t>(), A_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array.data_ptr<int64_t>(), B_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array.data_ptr<int64_t>(), C_array.data(), n1 * sizeof(float*), cudaMemcpyHostToDevice);
    
    cublasStatus_t status = cublasSgemmBatched(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        m2*r2, B, n2,
        &alpha,
        (const float**)d_B_array.data_ptr<int64_t>(), n2,
        (const float**)d_A_array.data_ptr<int64_t>(), n2,
        &beta,
        (float**)d_C_array.data_ptr<int64_t>(), m2*r2,
        n1
    );
    checkCublasStatus(status, "cublasSgemmBatched for grad_out1_3d");
}

// Batched GEMM wrapper for grad_W1 computation
static void batched_gemm_for_grad_W1(
    int B, int m2, int m1, int n1, int r2,
    torch::Tensor x_3d,
    torch::Tensor grad_out1,
    torch::Tensor grad_W1
) {
    // Implementation similar to batched_gemm_for_grad_W2 but for grad_W1 computation
    std::vector<const float*> A_array(m2);
    std::vector<const float*> B_array(m2);
    std::vector<float*> C_array(m2);
    
    auto d_A_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_B_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_C_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    for (int i = 0; i < m2; i++) {
        A_array[i] = x_3d.select(1, i).contiguous().data_ptr<float>();
        B_array[i] = grad_out1.select(1, i).contiguous().data_ptr<float>();
        C_array[i] = grad_W1.select(0, i).data_ptr<float>();
    }
    
    cudaMemcpy(d_A_array.data_ptr<int64_t>(), A_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array.data_ptr<int64_t>(), B_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array.data_ptr<int64_t>(), C_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    
    cublasStatus_t status = cublasSgemmBatched(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n1*r2, m1, B,
        &alpha,
        (const float**)d_B_array.data_ptr<int64_t>(), n1*r2,
        (const float**)d_A_array.data_ptr<int64_t>(), m1,
        &beta,
        (float**)d_C_array.data_ptr<int64_t>(), n1*r2,
        m2
    );
    checkCublasStatus(status, "cublasSgemmBatched for grad_W1");
}

// Batched GEMM wrapper for grad_x computation
static void batched_gemm_for_grad_x(
    int B, int m2, int m1, int n1, int r2,
    torch::Tensor grad_out1,
    torch::Tensor W1,
    torch::Tensor grad_x
) {
    // Implementation similar to batched_gemm_for_grad_out1_3d but for grad_x computation
    std::vector<const float*> A_array(m2);
    std::vector<const float*> B_array(m2);
    std::vector<float*> C_array(m2);
    
    auto d_A_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_B_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto d_C_array = torch::empty({m2}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    for (int i = 0; i < m2; i++) {
        A_array[i] = grad_out1.select(1, i).contiguous().data_ptr<float>();
        B_array[i] = W1.select(0, i).contiguous().reshape({m1, n1*r2}).data_ptr<float>();
        C_array[i] = grad_x.select(1, i).data_ptr<float>();
    }
    
    cudaMemcpy(d_A_array.data_ptr<int64_t>(), A_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array.data_ptr<int64_t>(), B_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array.data_ptr<int64_t>(), C_array.data(), m2 * sizeof(float*), cudaMemcpyHostToDevice);
    
    cublasStatus_t status = cublasSgemmBatched(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        m1, B, n1*r2,
        &alpha,
        (const float**)d_B_array.data_ptr<int64_t>(), n1*r2,
        (const float**)d_A_array.data_ptr<int64_t>(), n1*r2,
        &beta,
        (float**)d_C_array.data_ptr<int64_t>(), m1,
        m2
    );
    checkCublasStatus(status, "cublasSgemmBatched for grad_x");
}

// Backward pass with optimizations
std::vector<torch::Tensor> btt_cuda_backward_optimized(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    torch::Tensor out1,
    int m1, int m2,
    int n1, int n2,
    int r1, int r2
) {
    init_cuda();
    int B = input.size(0);
    
    // Ensure inputs are on GPU and contiguous
    grad_output = grad_output.contiguous();
    input = input.contiguous();
    W1 = W1.contiguous();
    W2 = W2.contiguous();
    out1 = out1.contiguous();
    
    // grad_output: (B,n1*n2) -> (B,n1,n2)
    auto grad_out2 = grad_output.reshape({B, n1, n2});
    auto out1_3d = out1.transpose(1, 2).contiguous().reshape({B, n1, m2*r2});
    
    // Pre-allocate tensors for batched operations
    auto grad_W2 = torch::zeros({n1, m2*r2, n2}, input.options());
    batched_gemm_for_grad_W2(B, n1, m2, r2, n2, out1_3d, grad_out2, grad_W2);
    
    // Pre-allocate for grad_out1_3d computation
    auto grad_out1_3d = torch::empty({B, n1, m2*r2}, input.options());
    batched_gemm_for_grad_out1_3d(B, n1, m2, r2, n2, grad_out2, W2, grad_out1_3d);
    
    // grad_out1 = reshape to (B,m2,n1*r2)
    auto grad_out1 = grad_out1_3d.reshape({B, n1, m2, r2}).transpose(1, 2).reshape({B, m2, n1*r2});
    
    // Pre-allocate for grad_W1 computation
    auto x_3d = input.reshape({B, m2, m1});
    auto grad_W1 = torch::empty({m2, m1, n1*r2}, input.options());
    batched_gemm_for_grad_W1(B, m2, m1, n1, r2, x_3d, grad_out1, grad_W1);
    
    // Pre-allocate for grad_x computation
    auto grad_x = torch::empty({B, m2, m1}, input.options());
    batched_gemm_for_grad_x(B, m2, m1, n1, r2, grad_out1, W1, grad_x);
    
    auto grad_input = grad_x.reshape({B, m1*m2});
    
    // Ensure all CUDA operations are complete
    cudaStreamSynchronize(stream);
    
    return {grad_input, grad_W1, grad_W2};
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "BTT CUDA with cuBLAS optimized (batched version)";
    
    m.def("forward", &btt_cuda_forward_optimized, "BTT forward optimized (CUDA cuBLAS)",
          py::arg("input"), py::arg("W1"), py::arg("W2"),
          py::arg("m1"), py::arg("m2"),
          py::arg("n1"), py::arg("n2"),
          py::arg("r1"), py::arg("r2")
    );
    
    m.def("backward", &btt_cuda_backward_optimized, "BTT backward optimized (CUDA cuBLAS)",
          py::arg("grad_output"), py::arg("input"),
          py::arg("W1"), py::arg("W2"), py::arg("out1"),
          py::arg("m1"), py::arg("m2"),
          py::arg("n1"), py::arg("n2"),
          py::arg("r1"), py::arg("r2")
    );
}