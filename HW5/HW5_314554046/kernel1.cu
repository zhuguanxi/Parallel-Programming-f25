#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

// check 
static inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

__device__ int mandel(float c_re, float c_im, int max_iter)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < max_iter; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandel_kernel(
    int *d_img,
    float lower_x, float lower_y,
    float step_x, float step_y,
    int res_x, int res_y,
    int max_iterations)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= res_x || py >= res_y)
        return;

    float x = lower_x + px * step_x;
    float y = lower_y + py * step_y;

    int idx = py * res_x + px;
    d_img[idx] = mandel(x, y, max_iterations);
}

void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    const size_t num_pixels = static_cast<size_t>(res_x) * res_y;
    const float step_x = (upper_x - lower_x) / static_cast<float>(res_x);
    const float step_y = (upper_y - lower_y) / static_cast<float>(res_y);

    int *h_buf = new int[num_pixels];

    // device buffer
    int *d_img = nullptr;
    checkCuda(cudaMalloc(&d_img, num_pixels * sizeof(int)), "cudaMalloc d_img");

    dim3 blockDim(16, 16);
    dim3 gridDim((res_x + blockDim.x - 1) / blockDim.x,
                 (res_y + blockDim.y - 1) / blockDim.y);

    mandel_kernel<<<gridDim, blockDim>>>(
        d_img,
        lower_x, lower_y,
        step_x, step_y,
        res_x, res_y,
        max_iterations);

    checkCuda(cudaGetLastError(), "mandel_kernel launch");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    checkCuda(cudaMemcpy(h_buf, d_img, num_pixels * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // 填回 img
    for (size_t i = 0; i < num_pixels; ++i)
        img[i] = h_buf[i];

    cudaFree(d_img);
    delete[] h_buf;
}

