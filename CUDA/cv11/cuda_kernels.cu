
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_img.h"


void launchKernel(void (*kernel)(CudaImg, CudaImg), CudaImg input_img, CudaImg output_img) {
    cudaError_t cerr;
    int block_size = 16;
    dim3 blocks((output_img.m_size.x + block_size - 1) / block_size, 
                (output_img.m_size.y + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);
    
    kernel<<<blocks, threads>>>(input_img, output_img);
    
    if ((cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));
    
    cudaDeviceSynchronize();
}


void launchFloatKernel(void (*kernel)(CudaImg, CudaImg, float), CudaImg input_img, CudaImg output_img, float param) {
    cudaError_t cerr;
    int block_size = 16;
    dim3 blocks((output_img.m_size.x + block_size - 1) / block_size, 
                (output_img.m_size.y + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);
    
    kernel<<<blocks, threads>>>(input_img, output_img, param);
    
    if ((cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));
    
    cudaDeviceSynchronize();
}


void launchAlphaBlendKernel(void (*kernel)(CudaImg, CudaImg, int, int), 
                           CudaImg background_img, CudaImg object_img, 
                           int x_offset, int y_offset) {
    cudaError_t cerr;
    int block_size = 16;
    dim3 blocks((object_img.m_size.x + block_size - 1) / block_size, 
                (object_img.m_size.y + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);
    
    kernel<<<blocks, threads>>>(background_img, object_img, x_offset, y_offset);
    
    if ((cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));
    
    cudaDeviceSynchronize();
}


__global__ void kernel_bilinear_scale(CudaImg input_img, CudaImg output_img) {
    int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (dst_x >= output_img.m_size.x || dst_y >= output_img.m_size.y)
        return;
    
    float scale_x = (float)input_img.m_size.x / output_img.m_size.x;
    float scale_y = (float)input_img.m_size.y / output_img.m_size.y;
    
    float src_x = dst_x * scale_x;
    float src_y = dst_y * scale_y;
    
    int x1 = floor(src_x);
    int y1 = floor(src_y);
    int x2 = min(x1 + 1, input_img.m_size.x - 1);
    int y2 = min(y1 + 1, input_img.m_size.y - 1);
    
    float x_frac = src_x - x1;
    float y_frac = src_y - y1;
    
    uchar4 p11 = input_img.m_p_uchar4[y1 * input_img.m_size.x + x1];
    uchar4 p12 = input_img.m_p_uchar4[y1 * input_img.m_size.x + x2];
    uchar4 p21 = input_img.m_p_uchar4[y2 * input_img.m_size.x + x1];
    uchar4 p22 = input_img.m_p_uchar4[y2 * input_img.m_size.x + x2];
    
    uchar4 result;
    result.x = (1 - x_frac) * (1 - y_frac) * p11.x + 
               x_frac * (1 - y_frac) * p12.x + 
               (1 - x_frac) * y_frac * p21.x + 
               x_frac * y_frac * p22.x;
    
    result.y = (1 - x_frac) * (1 - y_frac) * p11.y + 
               x_frac * (1 - y_frac) * p12.y + 
               (1 - x_frac) * y_frac * p21.y + 
               x_frac * y_frac * p22.y;
    
    result.z = (1 - x_frac) * (1 - y_frac) * p11.z + 
               x_frac * (1 - y_frac) * p12.z + 
               (1 - x_frac) * y_frac * p21.z + 
               x_frac * y_frac * p22.z;
    
    result.w = (1 - x_frac) * (1 - y_frac) * p11.w + 
               x_frac * (1 - y_frac) * p12.w + 
               (1 - x_frac) * y_frac * p21.w + 
               x_frac * y_frac * p22.w;
    
    output_img.m_p_uchar4[dst_y * output_img.m_size.x + dst_x] = result;
}


__global__ void kernel_rotate(CudaImg input_img, CudaImg output_img, float angle) {
    int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (dst_x >= output_img.m_size.x || dst_y >= output_img.m_size.y)
        return;
    
    // Calculate position relative to center
    float cx = dst_x - output_img.m_size.x / 2.0f;
    float cy = dst_y - output_img.m_size.y / 2.0f;
    
    // Rotate coordinates
    float sin_val = sinf(angle);
    float cos_val = cosf(angle);
    
    float src_cx = cos_val * cx - sin_val * cy;
    float src_cy = sin_val * cx + cos_val * cy;
    
    // Convert back to image coordinates
    float src_x = src_cx + input_img.m_size.x / 2.0f;
    float src_y = src_cy + input_img.m_size.y / 2.0f;
    
    // Check if the source coordinates are within bounds
    if (src_x < 0 || src_x >= input_img.m_size.x - 1 || 
        src_y < 0 || src_y >= input_img.m_size.y - 1) {
        // Out of bounds, set to transparent
        uchar4 transparent = {0, 0, 0, 0};
        output_img.m_p_uchar4[dst_y * output_img.m_size.x + dst_x] = transparent;
        return;
    }
    
    // Bilinear interpolation
    int x1 = floor(src_x);
    int y1 = floor(src_y);
    int x2 = min(x1 + 1, input_img.m_size.x - 1);
    int y2 = min(y1 + 1, input_img.m_size.y - 1);
    
    float x_frac = src_x - x1;
    float y_frac = src_y - y1;
    
    uchar4 p11 = input_img.m_p_uchar4[y1 * input_img.m_size.x + x1];
    uchar4 p12 = input_img.m_p_uchar4[y1 * input_img.m_size.x + x2];
    uchar4 p21 = input_img.m_p_uchar4[y2 * input_img.m_size.x + x1];
    uchar4 p22 = input_img.m_p_uchar4[y2 * input_img.m_size.x + x2];
    
    uchar4 result;
    result.x = (1 - x_frac) * (1 - y_frac) * p11.x + 
               x_frac * (1 - y_frac) * p12.x + 
               (1 - x_frac) * y_frac * p21.x + 
               x_frac * y_frac * p22.x;
    
    result.y = (1 - x_frac) * (1 - y_frac) * p11.y + 
               x_frac * (1 - y_frac) * p12.y + 
               (1 - x_frac) * y_frac * p21.y + 
               x_frac * y_frac * p22.y;
    
    result.z = (1 - x_frac) * (1 - y_frac) * p11.z + 
               x_frac * (1 - y_frac) * p12.z + 
               (1 - x_frac) * y_frac * p21.z + 
               x_frac * y_frac * p22.z;
    
    result.w = (1 - x_frac) * (1 - y_frac) * p11.w + 
               x_frac * (1 - y_frac) * p12.w + 
               (1 - x_frac) * y_frac * p21.w + 
               x_frac * y_frac * p22.w;
    
    output_img.m_p_uchar4[dst_y * output_img.m_size.x + dst_x] = result;
}


__global__ void kernel_motion_blur(CudaImg input_img, CudaImg output_img, float blur_level) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (x >= output_img.m_size.x || y >= output_img.m_size.y)
        return;
    
    if (x <= 0 || x >= input_img.m_size.x - 1 || y <= 0 || y >= input_img.m_size.y - 1) {
        output_img.m_p_uchar4[y * output_img.m_size.x + x] = input_img.m_p_uchar4[y * input_img.m_size.x + x];
        return;
    }
    
    // Pairs for 8-neighborhood
    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    
    uchar4 center = input_img.m_p_uchar4[y * input_img.m_size.x + x];
    
    // Start with the center pixel
    float sumR = center.x;
    float sumG = center.y;
    float sumB = center.z;
    float sumA = center.w;
    float weight = 1.0f;
    
    // Add weighted neighbors
    for (int i = 0; i < 8; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        uchar4 neighbor = input_img.m_p_uchar4[ny * input_img.m_size.x + nx];
        
        float factor = blur_level;
        
        sumR += neighbor.x * factor;
        sumG += neighbor.y * factor;
        sumB += neighbor.z * factor;
        sumA += neighbor.w * factor;
        weight += factor;
    }
    
    // Average
    uchar4 result;
    result.x = (unsigned char)(sumR / weight);
    result.y = (unsigned char)(sumG / weight);
    result.z = (unsigned char)(sumB / weight);
    result.w = (unsigned char)(sumA / weight);
    
    output_img.m_p_uchar4[y * output_img.m_size.x + x] = result;
}


__global__ void kernel_alpha_blend_insert(CudaImg background_img, CudaImg object_img, 
                                          int x_offset, int y_offset) {
    int obj_x = blockDim.x * blockIdx.x + threadIdx.x;
    int obj_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (obj_x >= object_img.m_size.x || obj_y >= object_img.m_size.y)
        return;
    
    int bg_x = obj_x + x_offset;
    int bg_y = obj_y + y_offset;
    
    if (bg_x < 0 || bg_x >= background_img.m_size.x || 
        bg_y < 0 || bg_y >= background_img.m_size.y)
        return;
    
    uchar4 obj_pixel = object_img.m_p_uchar4[obj_y * object_img.m_size.x + obj_x];
    
    if (obj_pixel.w == 0)
        return;
    
    uchar3 bg_pixel = background_img.m_p_uchar3[bg_y * background_img.m_size.x + bg_x];
    
    float alpha = obj_pixel.w / 255.0f;
    
    uchar3 result;
    result.x = alpha * obj_pixel.x + (1.0f - alpha) * bg_pixel.x;
    result.y = alpha * obj_pixel.y + (1.0f - alpha) * bg_pixel.y;
    result.z = alpha * obj_pixel.z + (1.0f - alpha) * bg_pixel.z;
    
    background_img.m_p_uchar3[bg_y * background_img.m_size.x + bg_x] = result;
}


void cu_bilinear_scale(CudaImg &input_img, CudaImg &output_img) {
    launchKernel(kernel_bilinear_scale, input_img, output_img);
}

void cu_rotate(CudaImg &input_img, CudaImg &output_img, float angle) {
    launchFloatKernel(kernel_rotate, input_img, output_img, angle);
}

void cu_motion_blur(CudaImg &input_img, CudaImg &output_img, float blur_level) {
    launchFloatKernel(kernel_motion_blur, input_img, output_img, blur_level);
}

void cu_alpha_blend_insert(CudaImg &background_img, CudaImg &object_img, int x_offset, int y_offset) {
    launchAlphaBlendKernel(kernel_alpha_blend_insert, background_img, object_img, x_offset, y_offset);
}
