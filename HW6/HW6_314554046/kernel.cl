__kernel void convolution(
    __global const float *inputImage,
    __constant const float *filter,
    __global float *outputImage,
    const int image_Height,
    const int image_Width,
    const int filterWidth) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halffilter_size = filterWidth / 2;
    float sum;
    int k, l;

    sum = 0;
    for (k = -halffilter_size; k <= halffilter_size; k++)
    {
        for (l = -halffilter_size; l <= halffilter_size; l++)
        {
            if (y + k >= 0 && y + k < image_Height &&
                x + l >= 0 && x + l < image_Width)
            {
                sum += inputImage[(y + k) * image_Width + x + l] *
                        filter[(k + halffilter_size) * filterWidth +
                                l + halffilter_size];
            }
        }
    }
    outputImage[y * image_Width + x] = sum;
}
