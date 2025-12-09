// ===== kernel3.cu =====
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

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

__global__ void mandel_kernel_group(
    int *d_img, size_t pitch,
    float lower_x, float lower_y,
    float step_x, float step_y,
    int res_x, int res_y,
    int max_iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = res_x * res_y;

    for (int idx = tid; idx < total; idx += stride)
    {
        int px = idx % res_x;
        int py = idx / res_x;

        float x = lower_x + px * step_x;
        float y = lower_y + py * step_y;

        int value = mandel(x, y, max_iterations);

        int *row = (int *)((char *)d_img + py * pitch);
        row[px] = value;
    }
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

    int *h_buf = nullptr;
    checkCuda(cudaHostAlloc((void **)&h_buf,
                            num_pixels * sizeof(int),
                            cudaHostAllocDefault),
              "cudaHostAlloc h_buf");

    int *d_img = nullptr;
    size_t pitch = 0;
    checkCuda(cudaMallocPitch((void **)&d_img,
                              &pitch,
                              res_x * sizeof(int),
                              res_y),
              "cudaMallocPitch d_img");

    int threadsPerBlock = 128;
    int blocks = (static_cast<int>(num_pixels) + threadsPerBlock - 1) / threadsPerBlock;

    mandel_kernel_group<<<blocks, threadsPerBlock>>>(
        d_img, pitch,
        lower_x, lower_y,
        step_x, step_y,
        res_x, res_y,
        max_iterations);

    checkCuda(cudaGetLastError(), "mandel_kernel_group launch");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    /*
    for (int y = 0; y < res_y; ++y)
    {
        int *dst = h_buf + static_cast<size_t>(y) * res_x;
        int *src = (int *)((char *)d_img + y * pitch);
        checkCuda(cudaMemcpy(dst, src,
                             res_x * sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy row D2H");
    }
    */

    checkCuda(cudaMemcpy2D(
              h_buf,
              res_x * sizeof(int),
              d_img,
              pitch,
              res_x * sizeof(int),
              res_y,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy2D D2H");

    for (size_t i = 0; i < num_pixels; ++i)
        img[i] = h_buf[i];

    cudaFree(d_img);
    cudaFreeHost(h_buf);
}

