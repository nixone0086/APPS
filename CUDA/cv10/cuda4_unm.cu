#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_img.h"

inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA Error at %s:%d - %s\n", file, line, cudaGetErrorString(err));
    }
}
#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

//1
__global__ void kernel_bilinear_scale(CudaImg input_img, CudaImg output_img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_img.m_size.x || y >= output_img.m_size.y)
        return;
    float x_ratio = (float)(input_img.m_size.x - 1) / (float)(output_img.m_size.x - 1);
    float y_ratio = (float)(input_img.m_size.y - 1) / (float)(output_img.m_size.y - 1);
    
    float src_x = x * x_ratio;
    float src_y = y * y_ratio;
    
    int x1 = (int)src_x;
    int y1 = (int)src_y;
    int x2 = min(x1 + 1, input_img.m_size.x - 1);
    int y2 = min(y1 + 1, input_img.m_size.y - 1);
    
    float x_diff = src_x - x1;
    float y_diff = src_y - y1;
    
    //get four neighboring pixels
    uchar4 p1 = input_img.at_rgba(x1, y1);
    uchar4 p2 = input_img.at_rgba(x2, y1);
    uchar4 p3 = input_img.at_rgba(x1, y2);
    uchar4 p4 = input_img.at_rgba(x2, y2);
    
    //perform bilinear interpolation for each channel
    uchar4 result;
    result.x = (unsigned char)((1 - x_diff) * (1 - y_diff) * p1.x + 
                               x_diff * (1 - y_diff) * p2.x + 
                               (1 - x_diff) * y_diff * p3.x + 
                               x_diff * y_diff * p4.x);
    
    result.y = (unsigned char)((1 - x_diff) * (1 - y_diff) * p1.y + 
                               x_diff * (1 - y_diff) * p2.y + 
                               (1 - x_diff) * y_diff * p3.y + 
                               x_diff * y_diff * p4.y);
    
    result.z = (unsigned char)((1 - x_diff) * (1 - y_diff) * p1.z + 
                               x_diff * (1 - y_diff) * p2.z + 
                               (1 - x_diff) * y_diff * p3.z + 
                               x_diff * y_diff * p4.z);
    
    result.w = (unsigned char)((1 - x_diff) * (1 - y_diff) * p1.w + 
                               x_diff * (1 - y_diff) * p2.w + 
                               (1 - x_diff) * y_diff * p3.w + 
                               x_diff * y_diff * p4.w);
    
    output_img.set_rgba(x, y, result);
}

//2
__global__ void kernel_swap_channels(CudaImg input_img, CudaImg output_img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= input_img.m_size.x || y >= input_img.m_size.y)
        return;
    
    uchar4 pixel = input_img.at_rgba(x, y);
    
    uchar4 swapped;
    swapped.x = pixel.z; // B becomes R
    swapped.y = pixel.y; // G stays G
    swapped.z = pixel.x; // R becomes B
    swapped.w = pixel.w; // Alpha stays the same
    
    output_img.set_rgba(x, y, swapped);
}
//3
__global__ void kernel_rotate90(CudaImg input_img, CudaImg output_img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_img.m_size.x || y >= output_img.m_size.y)
        return;
    int src_x = input_img.m_size.y - 1 - y;
    int src_y = x;
    
    if (src_x >= 0 && src_x < input_img.m_size.x && 
        src_y >= 0 && src_y < input_img.m_size.y) {
        uchar4 pixel = input_img.at_rgba(src_x, src_y);
        output_img.set_rgba(x, y, pixel);
    }
}


//4
__global__ void kernel_to_grayscale(CudaImg input_img, CudaImg output_img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= input_img.m_size.x || y >= input_img.m_size.y)
        return;
    
    uchar4 pixel = input_img.at_rgba(x, y);
    
    //convert RGB to grayscale using standard luminance formula
    unsigned char gray = (unsigned char)(0.114f * pixel.x + 0.587f * pixel.y + 0.299f * pixel.z);
    
    uchar4 result;
    result.x = gray;
    result.y = gray;
    result.z = gray;
    result.w = pixel.w; //keep alpha channel
    
    output_img.set_rgba(x, y, result);
}

//5
__global__ void kernel_alpha_blend_insert(CudaImg background_img, CudaImg object_img, int x_offset, int y_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= object_img.m_size.x || y >= object_img.m_size.y)
        return;
    
    int bg_x = x + x_offset;
    int bg_y = y + y_offset;

    if (bg_x < 0 || bg_x >= background_img.m_size.x || 
        bg_y < 0 || bg_y >= background_img.m_size.y)
        return;
    
    uchar4 obj_pixel = object_img.at_rgba(x, y);
    uchar3 bg_pixel = background_img.at_color(bg_x, bg_y);
    
    float alpha = obj_pixel.w / 255.0f;
    
    uchar3 result;
    result.x = (unsigned char)(alpha * obj_pixel.x + (1 - alpha) * bg_pixel.x);
    result.y = (unsigned char)(alpha * obj_pixel.y + (1 - alpha) * bg_pixel.y);
    result.z = (unsigned char)(alpha * obj_pixel.z + (1 - alpha) * bg_pixel.z);
    
    background_img.set_color(bg_x, bg_y, result);
}


extern "C" void cu_bilinear_scale(CudaImg input_img, CudaImg output_img)
{
    int block_size = 16;
    dim3 blocks((output_img.m_size.x + block_size - 1) / block_size,
                (output_img.m_size.y + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);
    
    kernel_bilinear_scale<<<blocks, threads>>>(input_img, output_img);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

extern "C" void cu_swap_channels(CudaImg input_img, CudaImg output_img)
{
    int block_size = 16;
    dim3 blocks((input_img.m_size.x + block_size - 1) / block_size,
                (input_img.m_size.y + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);
    
    kernel_swap_channels<<<blocks, threads>>>(input_img, output_img);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

extern "C" void cu_rotate90(CudaImg input_img, CudaImg output_img)
{
    int block_size = 16;
    dim3 blocks((output_img.m_size.x + block_size - 1) / block_size,
                (output_img.m_size.y + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);
    
    kernel_rotate90<<<blocks, threads>>>(input_img, output_img);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

extern "C" void cu_to_grayscale(CudaImg input_img, CudaImg output_img)
{
    int block_size = 16;
    dim3 blocks((input_img.m_size.x + block_size - 1) / block_size,
                (input_img.m_size.y + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);
    
    kernel_to_grayscale<<<blocks, threads>>>(input_img, output_img);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

extern "C" void cu_alpha_blend_insert(CudaImg background_img, CudaImg object_img, int x_offset, int y_offset)
{
    int block_size = 16;
    dim3 blocks((object_img.m_size.x + block_size - 1) / block_size,
                (object_img.m_size.y + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);
    
    kernel_alpha_blend_insert<<<blocks, threads>>>(background_img, object_img, x_offset, y_offset);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}
