#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


#define TILE_SIZE 16

// Forward kernel
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

    const int batch_idx = blockIdx.x;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.z * TILE_SIZE + threadIdx.x;

    if (batch_idx >= batch_size) return;

    // First multiply
    if (row < m2 && col < n1 * r2) {
        scalar_t sum = 0;
        for (int k = 0; k < m1; k++) {
            const int input_idx = batch_idx * m1 * m2 + row * m1 + k;
            const int w1_idx = row * (m1 * r1) * (n1 * r2) + (k * r1) * (n1 * r2) + col;
            for (int r = 0; r < r1; r++) {
                sum += input[input_idx] * W1[w1_idx + r * (n1 * r2)];
            }
        }
        intermediate[batch_idx * m2 * n1 * r2 + row * n1 * r2 + col] = sum;
    }

    __syncthreads();

    // Second multiply
    if (row < n1 && col < n2) {
        scalar_t sum = 0;
        for (int k = 0; k < m2; k++) {
            for (int r = 0; r < r2; r++) {
                const int interm_idx = batch_idx * m2 * n1 * r2 + k * n1 * r2 + row * r2 + r;
                const int w2_idx = row * (m2 * r2) * n2 + (k * r2 + r) * n2 + col;
                sum += intermediate[interm_idx] * W2[w2_idx];
            }
        }
        output[batch_idx * n1 * n2 + row * n2 + col] = sum;
    }
}

// Backward kernels
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

    const int batch_idx = blockIdx.x;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.z * TILE_SIZE + threadIdx.x;

    if (batch_idx >= batch_size) return;

    if (row < m1 && col < m2) {
        scalar_t sum = 0;
        for (int k = 0; k < n1; k++) {
            for (int r = 0; r < r2; r++) {
                const int interm_grad = intermediate[batch_idx * m2 * n1 * r2 + col * n1 * r2 + k * r2 + r];
                const int w1_idx = col * (m1 * r1) * (n1 * r2) + (row * r1) * (n1 * r2) + k * r2 + r;
                sum += interm_grad * W1[w1_idx];
            }
        }
        grad_input[batch_idx * m1 * m2 + col * m1 + row] = sum;
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

    // One thread per W1/W2 element
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.z * TILE_SIZE + threadIdx.x;

    // Gradient for W2
    if (row < n1 && col < n2) {
        scalar_t sum = 0;
        for (int b = 0; b < batch_size; b++) {
            const scalar_t grad = grad_output[b * n1 * n2 + row * n2 + col];
            for (int k = 0; k < m2; k++) {
                for (int r = 0; r < r2; r++) {
                    const int interm_idx = b * m2 * n1 * r2 + k * n1 * r2 + row * r2 + r;
                    sum += intermediate[interm_idx] * grad;
                }
            }
        }
        grad_W2[row * (m2 * r2) * n2 + col] = sum;
    }

    // Gradient for W1
    if (row < m2 && col < n1 * r2) {
        scalar_t sum = 0;
        for (int b = 0; b < batch_size; b++) {
            const scalar_t interm_grad = intermediate[b * m2 * n1 * r2 + row * n1 * r2 + col];
            for (int k = 0; k < m1; k++) {
                const int input_idx = b * m1 * m2 + row * m1 + k;
                sum += input[input_idx] * interm_grad;
            }
        }
        grad_W1[row * (m1 * r1) * (n1 * r2) + col] = sum;
    }
}

torch::Tensor btt_cuda_forward(
    torch::Tensor input,
    torch::Tensor W1,
    torch::Tensor W2,
    const int m1, const int m2,
    const int n1, const int n2,
    const int r1, const int r2) {
    
    const int batch_size = input.size(0);
    auto output = torch::empty({batch_size, n1 * n2}, input.options());
    auto intermediate = torch::empty({batch_size, m2 * n1 * r2}, input.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        batch_size,
        (std::max(m2, n1) + TILE_SIZE - 1) / TILE_SIZE,
        (std::max(n1*r2, n2) + TILE_SIZE - 1) / TILE_SIZE
    );

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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    dim3 blocks_data(
        batch_size,
        (m1 + TILE_SIZE - 1) / TILE_SIZE,
        (m2 + TILE_SIZE - 1) / TILE_SIZE
    );

    dim3 blocks_weights(
        1,
        (std::max(m2, n1) + TILE_SIZE - 1) / TILE_SIZE,
        (std::max(n1*r2, n2) + TILE_SIZE - 1) / TILE_SIZE
    );

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "btt_backward_cuda", ([&] {
        btt_backward_data_kernel<scalar_t><<<blocks_data, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            W1.data_ptr<scalar_t>(),
            W2.data_ptr<scalar_t>(),
            intermediate.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            batch_size, m1, m2, n1, n2, r1, r2
        );

        btt_backward_weights_kernel<scalar_t><<<blocks_weights, threads>>>(
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