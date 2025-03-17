#include <torch/extension.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Global handle and streams for concurrent operations
static cublasHandle_t cublas_handle;
static cudaStream_t streams[2];  // Multiple streams for overlapping operations
static bool initialized = false;

inline void checkCublasStatus(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS error at: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCudaStatus(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error at: " << msg << ": " 
                  << cudaGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void initialize() {
    if (!initialized) {
        checkCublasStatus(cublasCreate(&cublas_handle), "cublasCreate");
        
        // Create CUDA streams for concurrent operations
        for (int i = 0; i < 2; i++) {
            checkCudaStatus(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
        }
        
        initialized = true;
    }
}

void cleanup() {
    if (initialized) {
        checkCublasStatus(cublasDestroy(cublas_handle), "cublasDestroy");
        
        for (int i = 0; i < 2; i++) {
            checkCudaStatus(cudaStreamDestroy(streams[i]), "cudaStreamDestroy");
        }
        
        initialized = false;
    }
}

// Optimized GEMM for multiple matrices (avoids loop overhead)
// Efficiently computes multiple GEMM operations by reducing API calls
void optimized_gemm_batched(
    int B, int slices, int m, int k, int n,
    const float* input_data, int input_stride,
    const float* weight_data, int weight_stride,
    float* output_data, int output_stride
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Use multi-stream approach for concurrent computation
    for (int i = 0; i < slices; i++) {
        // Select streams in round-robin fashion for load balancing
        cudaStream_t stream = streams[i % 2];
        checkCublasStatus(cublasSetStream(cublas_handle, stream), "cublasSetStream");
        
        // Calculate pointers for this slice
        const float* input_slice = input_data + i * input_stride;
        const float* weight_slice = weight_data + i * weight_stride;
        float* output_slice = output_data + i * output_stride;
        
        // Execute GEMM for this slice
        checkCublasStatus(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,  // Use transposed operations to handle row-major
            n, B, k,                   // Dimensions for the operation
            &alpha,
            weight_slice, k,           // Weight matrix (transposed)
            input_slice, k,            // Input matrix
            &beta,
            output_slice, n            // Output matrix
        ), "cublasSgemm in optimized_gemm_batched");
    }
    
    // Synchronize all streams to ensure computation is complete
    for (int i = 0; i < 2; i++) {
        checkCudaStatus(cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize");
    }
}

// Optimized forward pass
std::pair<torch::Tensor, torch::Tensor> btt_cuda_forward(
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    int m1, int m2,
    int n1, int n2,
    int r1, int r2
) {
    initialize();
    int B = input.size(0);
    
    // Reshape input to 3D without copy
    auto x_3d = input.reshape({B, m2, m1});
    
    // Allocate memory for intermediate results
    auto out1 = torch::empty({B, m2, n1*r2}, input.options());
    
    // First layer gemm: loop optimization
    optimized_gemm_batched(
        B, m2, m1, m1, n1*r2,
        x_3d.data_ptr<float>(), m1,
        W1.reshape({m2, m1, n1*r2}).data_ptr<float>(), m1 * (n1*r2),
        out1.data_ptr<float>(), n1*r2
    );
    
    // Use torch operations for reshape and transpose (more reliable than custom kernel)
    auto out1_3d = out1.transpose(1, 2).reshape({B, n1, m2*r2}).contiguous();
    
    // Allocate memory for output
    auto out2 = torch::empty({B, n1, n2}, input.options());
    
    // Second layer gemm: loop optimization
    optimized_gemm_batched(
        B, n1, m2*r2, m2*r2, n2,
        out1_3d.data_ptr<float>(), m2*r2,
        W2.reshape({n1, m2*r2, n2}).data_ptr<float>(), (m2*r2) * n2,
        out2.data_ptr<float>(), n2
    );
    
    // Reshape output to final dimensions
    auto output = out2.reshape({B, n1*n2});
    
    return {output, out1};
}

// Optimized backward pass
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
    initialize();
    int B = input.size(0);
    
    // Reshape grad_output to 3D
    auto grad_out2 = grad_output.reshape({B, n1, n2});
    
    // Use torch operations for reshape and transpose
    auto out1_3d = out1.transpose(1, 2).reshape({B, n1, m2*r2}).contiguous();
    auto x_3d = input.reshape({B, m2, m1});
    
    // Initialize gradient tensors
    auto grad_W2 = torch::zeros({n1, m2*r2, n2}, input.options());
    auto grad_out1_3d = torch::zeros({B, n1, m2*r2}, input.options());
    
    // Compute grad_W2 and grad_out1_3d in parallel using multi-stream
    for (int i = 0; i < n1; i++) {
        // Use stream 0 for grad_W2 computation
        checkCublasStatus(cublasSetStream(cublas_handle, streams[0]), "cublasSetStream");
        
        // grad_W2[i] = out1_3d[i].transpose(0,1) @ grad_out2[i]
        auto out1_3d_i = out1_3d.select(1, i).contiguous();
        auto grad_out2_i = grad_out2.select(1, i).contiguous();
        auto grad_W2_i = grad_W2.select(0, i).contiguous();
        
        float alpha = 1.0f;
        float beta = 0.0f;
        
        checkCublasStatus(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n2, m2*r2, B,
            &alpha,
            grad_out2_i.data_ptr<float>(), n2,
            out1_3d_i.data_ptr<float>(), m2*r2,
            &beta,
            grad_W2_i.data_ptr<float>(), n2
        ), "cublasSgemm for grad_W2");
        
        // Use stream 1 for grad_out1_3d computation
        checkCublasStatus(cublasSetStream(cublas_handle, streams[1]), "cublasSetStream");
        
        // grad_out1_3d[i] = grad_out2[i] @ W2[i].transpose(0,1)
        auto W2_i = W2.select(0, i).reshape({m2*r2, n2}).contiguous();
        auto grad_out1_3d_i = grad_out1_3d.select(1, i).contiguous();
        
        checkCublasStatus(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            m2*r2, B, n2,
            &alpha,
            W2_i.data_ptr<float>(), n2,
            grad_out2_i.data_ptr<float>(), n2,
            &beta,
            grad_out1_3d_i.data_ptr<float>(), m2*r2
        ), "cublasSgemm for grad_out1_3d");
    }
    
    // Synchronize streams before continuing
    for (int i = 0; i < 2; i++) {
        checkCudaStatus(cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize");
    }
    
    // Reshape and transpose grad_out1_3d to grad_out1
    auto grad_out1 = grad_out1_3d.reshape({B, n1, m2, r2}).transpose(1, 2).reshape({B, m2, n1*r2}).contiguous();
    
    // Compute grad_W1 and grad_input
    auto grad_W1 = torch::zeros({m2, m1, n1*r2}, input.options());
    auto grad_x = torch::zeros({B, m2, m1}, input.options());
    
    // Compute grad_W1 and grad_input in parallel using multi-stream
    for (int i = 0; i < m2; i++) {
        // Use stream 0 for grad_W1 computation
        checkCublasStatus(cublasSetStream(cublas_handle, streams[0]), "cublasSetStream");
        
        // grad_W1[i] = x_3d[i].transpose(0,1) @ grad_out1[i]
        auto x_slice = x_3d.select(1, i).contiguous();
        auto grad_out1_i = grad_out1.select(1, i).contiguous();
        auto grad_W1_i = grad_W1.select(0, i).contiguous();
        
        float alpha = 1.0f;
        float beta = 0.0f;
        
        checkCublasStatus(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n1*r2, m1, B,
            &alpha,
            grad_out1_i.data_ptr<float>(), n1*r2,
            x_slice.data_ptr<float>(), m1,
            &beta,
            grad_W1_i.data_ptr<float>(), n1*r2
        ), "cublasSgemm for grad_W1");
        
        // Use stream 1 for grad_x computation
        checkCublasStatus(cublasSetStream(cublas_handle, streams[1]), "cublasSetStream");
        
        // grad_x[i] = grad_out1[i] @ W1[i].transpose(0,1)
        auto W1_i = W1.select(0, i).reshape({m1, n1*r2}).contiguous();
        auto grad_x_i = grad_x.select(1, i).contiguous();
        
        checkCublasStatus(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            m1, B, n1*r2,
            &alpha,
            W1_i.data_ptr<float>(), n1*r2,
            grad_out1_i.data_ptr<float>(), n1*r2,
            &beta,
            grad_x_i.data_ptr<float>(), m1
        ), "cublasSgemm for grad_x");
    }
    
    // Synchronize streams before returning
    for (int i = 0; i < 2; i++) {
        checkCudaStatus(cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize");
    }
    
    // Reshape grad_x to match input shape
    auto grad_input = grad_x.reshape({B, m1*m2});
    
    return {grad_input, grad_W1, grad_W2};
}