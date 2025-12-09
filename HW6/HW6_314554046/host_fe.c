#include <stdio.h>
#include <stdlib.h>
#include "host_fe.h"
#include "helper.h"

void host_fe(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize  = imageHeight * imageWidth;

    static cl_command_queue commandQueue = NULL;
    static cl_kernel kernel = NULL;
    static cl_mem inputImg = NULL;
    static cl_mem filterMemObj = NULL;
    static cl_mem outputImg = NULL;

    static int prevImageHeight  = 0;
    static int prevImageWidth   = 0;
    static int prevFilterWidth  = 0;

    if (commandQueue == NULL)
    {
        commandQueue = clCreateCommandQueue(*context, *device, 0, &status);
//        CHECK(status, "clCreateCommandQueue");
    }

    if (kernel == NULL)
    {
        kernel = clCreateKernel(*program, "convolution", &status);
//        CHECK(status, "clCreateKernel");
    }

    if (inputImg == NULL ||
        outputImg == NULL ||
        prevImageHeight != imageHeight ||
        prevImageWidth  != imageWidth)
    {
        if (inputImg)  clReleaseMemObject(inputImg);
        if (outputImg) clReleaseMemObject(outputImg);

        inputImg = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageSize * sizeof(float), inputImage, &status);
//        CHECK(status, "clCreateBuffer");

        outputImg = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, imageSize * sizeof(float), outputImage, &status);
//        CHECK(status, "clCreateBuffer");

        prevImageHeight = imageHeight;
        prevImageWidth  = imageWidth;
    }

    if (filterMemObj == NULL || prevFilterWidth != filterWidth)
    {
        if (filterMemObj) clReleaseMemObject(filterMemObj);

        filterMemObj = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filterSize * sizeof(float), filter, &status);
        CHECK(status, "clCreateBuffer");

        prevFilterWidth = filterWidth;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputImg);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filterMemObj);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputImg);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&filterWidth);

    size_t globalWorkSize[2] = {imageWidth, imageHeight};
//  size_t localWorkSize[2] = {8, 8}; 

//    status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

    CHECK(status, "clEnqueueNDRangeKernel");
    
    status = clEnqueueReadBuffer(commandQueue, outputImg, CL_TRUE, 0, imageSize * sizeof(float), outputImage, 0, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer");
}

