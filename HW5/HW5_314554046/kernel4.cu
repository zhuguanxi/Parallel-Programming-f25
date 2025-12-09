#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float c_lower_x;
__constant__ float c_lower_y;
__constant__ float c_step_x;
__constant__ float c_step_y;
__constant__ int c_max_iterations;

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

__device__ __forceinline__ int mandel(float c_re, float c_im)
{
    float z_re = c_re, z_im = c_im;
    
    float z_re2 = z_re * z_re;
    float z_im2 = z_im * z_im;
    
    int i = 0;
    
    int max_iter = c_max_iterations;
    int unroll_limit = (max_iter / 8) * 8;
    
    for (; i < unroll_limit; i += 8)
    {
        if (z_re2 + z_im2 > 4.f) break;
        float new_im = 2.f * z_re * z_im;
        float new_re = z_re2 - z_im2;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
        
        if (z_re2 + z_im2 > 4.f) { i++; break; }
        new_im = 2.f * z_re * z_im;
        new_re = z_re2 - z_im2;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
        
        if (z_re2 + z_im2 > 4.f) { i += 2; break; }
        new_im = 2.f * z_re * z_im;
        new_re = z_re2 - z_im2;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
        
        if (z_re2 + z_im2 > 4.f) { i += 3; break; }
        new_im = 2.f * z_re * z_im;
        new_re = z_re2 - z_im2;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;

        if (z_re2 + z_im2 > 4.f) { i += 4; break; }
        new_im = 2.f * z_re * z_im;
        new_re = z_re2 - z_im2;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;

        if (z_re2 + z_im2 > 4.f) { i += 5; break; }
        new_im = 2.f * z_re * z_im;
        new_re = z_re2 - z_im2;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;

        if (z_re2 + z_im2 > 4.f) { i += 6; break; }
        new_im = 2.f * z_re * z_im;
        new_re = z_re2 - z_im2;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;

        if (z_re2 + z_im2 > 4.f) { i += 7; break; }
        new_im = 2.f * z_re * z_im;
        new_re = z_re2 - z_im2;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
    }

    for (; i < max_iter; ++i)
    {
        if (z_re2 + z_im2 > 4.f) break;
            float new_im = 2.f * z_re * z_im;
            float new_re = z_re2 - z_im2;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
            z_re2 = z_re * z_re;
            z_im2 = z_im * z_im;
    }
    
    return i;
}

__global__ void mandel_kernel(int *d_img, int res_x, int res_y)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= res_x || py >= res_y)
        return;

    float x = c_lower_x + px * c_step_x;
    float y = c_lower_y + py * c_step_y;

    int idx = py * res_x + px;
    d_img[idx] = mandel(x, y);
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

    static int *d_img = nullptr;
    static size_t cached_size = 0;
    
    if (d_img == nullptr || cached_size < num_pixels)
    {
        if (d_img != nullptr)
            cudaFree(d_img);
        
        cudaMalloc(&d_img, num_pixels * sizeof(int));
        cached_size = num_pixels;
    }

    cudaMemcpyToSymbol(c_lower_x, &lower_x, sizeof(float));
    cudaMemcpyToSymbol(c_lower_y, &lower_y, sizeof(float));
    cudaMemcpyToSymbol(c_step_x, &step_x, sizeof(float));
    cudaMemcpyToSymbol(c_step_y, &step_y, sizeof(float));
    cudaMemcpyToSymbol(c_max_iterations, &max_iterations, sizeof(int));

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((res_x + blockDim.x - 1) / blockDim.x,
                 (res_y + blockDim.y - 1) / blockDim.y);

    mandel_kernel<<<gridDim, blockDim>>>(d_img, res_x, res_y);

    cudaDeviceSynchronize();

    cudaMemcpy(img, d_img, num_pixels * sizeof(int), cudaMemcpyDeviceToHost);
}

