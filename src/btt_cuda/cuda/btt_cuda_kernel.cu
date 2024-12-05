// btt_cuda_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA forward kernel
template <typename scalar_t>
__global__ void btt_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ W1,
    const scalar_t* __restrict__ W2,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ intermediate,
    const int batch_size,
    const int m1, const int m2,
    const int n1, const int n2,
    const int r1, const int r2) {

    // Get batch index
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // First transformation
    for (int i = 0; i < m2; i++) {
        for (int j = 0; j < n1 * r2; j++) {
            scalar_t sum = 0;
            for (int k = 0; k < m1 * r1; k++) {
                int input_idx = batch_idx * m1 * m2 + i * m1 + k / r1;
                int w1_idx = i * (m1 * r1) * (n1 * r2) + k * (n1 * r2) + j;
                sum += input[input_idx] * W1[w1_idx];
            }
            intermediate[batch_idx * m2 * n1 * r2 + i * n1 * r2 + j] = sum;
        }
    }

    // Second transformation
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            scalar_t sum = 0;
            for (int k = 0; k < m2 * r2; k++) {
                int interm_idx = batch_idx * m2 * n1 * r2 + (k / r2) * n1 * r2 + i * r2 + (k % r2);
                int w2_idx = i * (m2 * r2) * n2 + k * n2 + j;
                sum += intermediate[interm_idx] * W2[w2_idx];
            }
            output[batch_idx * n1 * n2 + i * n2 + j] = sum;
        }
    }
}

// CUDA backward kernels
template <typename scalar_t>
__global__ void btt_backward_data_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ W1,
    const scalar_t* __restrict__ W2,
    const scalar_t* __restrict__ intermediate,
    scalar_t* __restrict__ grad_input,
    const int batch_size,
    const int m1, const int m2,
    const int n1, const int n2,
    const int r1, const int r2) {

    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Gradient w.r.t input
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < m2; j++) {
            scalar_t sum = 0;
            for (int k = 0; k < n1 * r2; k++) {
                for (int r = 0; r < r1; r++) {
                    sum += W1[j * (m1 * r1) * (n1 * r2) + (i * r1 + r) * (n1 * r2) + k] *
                           grad_output[batch_idx * n1 * n2 + (k / r2) * n2] *
                           W2[(k / r2) * (m2 * r2) * n2 + (j * r2 + k % r2) * n2];
                }
            }
            grad_input[batch_idx * m1 * m2 + j * m1 + i] = sum;
        }
    }
}

template <typename scalar_t>
__global__ void btt_backward_weights_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ intermediate,
    scalar_t* __restrict__ grad_W1,
    scalar_t* __restrict__ grad_W2,
    const int batch_size,
    const int m1, const int m2,
    const int n1, const int n2,
    const int r1, const int r2) {

    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Gradient w.r.t W2
    for (int i = 0; i < n1; i++) {
        for (int k = 0; k < m2 * r2; k++) {
            for (int j = 0; j < n2; j++) {
                int interm_idx = batch_idx * m2 * n1 * r2 + (k / r2) * n1 * r2 + i * r2 + (k % r2);
                int grad_out_idx = batch_idx * n1 * n2 + i * n2 + j;
                atomicAdd(&grad_W2[i * (m2 * r2) * n2 + k * n2 + j],
                         intermediate[interm_idx] * grad_output[grad_out_idx]);
            }
        }
    }

    // Gradient w.r.t W1
    for (int i = 0; i < m2; i++) {
        for (int k = 0; k < m1 * r1; k++) {
            for (int j = 0; j < n1 * r2; j++) {
                int input_idx = batch_idx * m1 * m2 + i * m1 + k / r1;
                atomicAdd(&grad_W1[i * (m1 * r1) * (n1 * r2) + k * (n1 * r2) + j],
                         input[input_idx] * grad_output[batch_idx * n1 * n2 + j / r2 * n2]);
            }
        }
    }
}

// C++ interface functions
torch::Tensor btt_cuda_forward(
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    const int m1, const int m2,
    const int n1, const int n2,
    const int r1, const int r2) {
    
    const int batch_size = input.size(0);
    auto output = torch::empty({batch_size, n1 * n2}, input.options());
    auto intermediate = torch::empty({batch_size, m2, n1, r2}, input.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "btt_forward_cuda", ([&] {
        btt_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            W1.data_ptr<scalar_t>(),
            W2.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            intermediate.data_ptr<scalar_t>(),
            batch_size, m1, m2, n1, n2, r1, r2
        );
    }));

    return output;
}

std::vector<torch::Tensor> btt_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    torch::Tensor intermediate,
    const int m1, const int m2,
    const int n1, const int n2,
    const int r1, const int r2) {
    
    const int batch_size = input.size(0);
    auto grad_input = torch::zeros_like(input);
    auto grad_W1 = torch::zeros_like(W1);
    auto grad_W2 = torch::zeros_like(W2);

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "btt_backward_cuda", ([&] {
        btt_backward_data_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            W1.data_ptr<scalar_t>(),
            W2.data_ptr<scalar_t>(),
            intermediate.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            batch_size, m1, m2, n1, n2, r1, r2
        );

        btt_backward_weights_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            intermediate.data_ptr<scalar_t>(),
            grad_W1.data_ptr<scalar_t>(),
            grad_W2.data_ptr<scalar_t>(),
            batch_size, m1, m2, n1, n2, r1, r2
        );
    }));

    return {grad_input, grad_W1, grad_W2};
}