#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_img.h"


__global__ void kernel_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    // X,Y 
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_color_cuda_img.m_size.y) return;
    if (l_x >= t_color_cuda_img.m_size.x) return;

    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x];

   
    t_bw_cuda_img.m_p_uchar1[l_y * t_bw_cuda_img.m_size.x + l_x].x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
}

void cu_run_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_grayscale<<<l_blocks, l_threads>>>(t_color_cuda_img, t_bw_cuda_img);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_invert_horizontal(CudaImg t_bgr_cuda_img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= t_bgr_cuda_img.m_size.x / 2 || y >= t_bgr_cuda_img.m_size.y)
        return;

    //calculate index for current pixel
    int idx = y * t_bgr_cuda_img.m_size.x + x;
    
    //calculate mirror index in the right half
    int mirror_x = t_bgr_cuda_img.m_size.x - 1 - x;
    int idx_mirror = y * t_bgr_cuda_img.m_size.x + mirror_x;
    
    //swap pixels
    uchar3 tmp = t_bgr_cuda_img.m_p_uchar3[idx];
    t_bgr_cuda_img.m_p_uchar3[idx] = t_bgr_cuda_img.m_p_uchar3[idx_mirror];
    t_bgr_cuda_img.m_p_uchar3[idx_mirror] = tmp;
}

__global__ void kernel_invert_vertical(CudaImg t_bgr_cuda_img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= t_bgr_cuda_img.m_size.x || y >= t_bgr_cuda_img.m_size.y / 2)
        return;

    int idx = y * t_bgr_cuda_img.m_size.x + x;

    int mirror_y = t_bgr_cuda_img.m_size.y - 1 - y;
    int idx_mirror = mirror_y * t_bgr_cuda_img.m_size.x + x;
    
    uchar3 tmp = t_bgr_cuda_img.m_p_uchar3[idx];
    t_bgr_cuda_img.m_p_uchar3[idx] = t_bgr_cuda_img.m_p_uchar3[idx_mirror];
    t_bgr_cuda_img.m_p_uchar3[idx_mirror] = tmp;
}

//emoving color as per original requirements
__global__ void kernel_remove_color_amount(CudaImg t_bgr_cuda_img, unsigned char r, unsigned char g, unsigned char b, float amount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= t_bgr_cuda_img.m_size.x || y >= t_bgr_cuda_img.m_size.y)
        return;

    int idx = y * t_bgr_cuda_img.m_size.x + x;
    uchar3 pixel = t_bgr_cuda_img.m_p_uchar3[idx];
    
    if (pixel.y > 90 ) {
        uchar3 black_pixel;
        black_pixel.x = 0; // Blue
        black_pixel.y = 0; // Green
        black_pixel.z = 0; // Red
        t_bgr_cuda_img.m_p_uchar3[idx] = black_pixel;
    }
}


// 1 color filter kernel
__global__ void kernel_color_filter(CudaImg t_input_img, CudaImg t_output_img, int color_channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= t_input_img.m_size.x || y >= t_input_img.m_size.y)
        return;

    int idx = y * t_input_img.m_size.x + x;
    uchar3 pixel = t_input_img.m_p_uchar3[idx];
    uchar3 filtered_pixel = make_uchar3(0, 0, 0);
    
    if (color_channel == 0) filtered_pixel.x = pixel.x;  // Blue
    else if (color_channel == 1) filtered_pixel.y = pixel.y;  // Green
    else if (color_channel == 2) filtered_pixel.z = pixel.z;  // Red
    
    t_output_img.m_p_uchar3[idx] = filtered_pixel;
}

//2 horizontal mirror kernel
__global__ void kernel_mirror_horizontal_full(CudaImg t_input_img, CudaImg t_output_img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= t_input_img.m_size.x || y >= t_input_img.m_size.y)
        return;

    int idx_src = y * t_input_img.m_size.x + x;
    
    //mirrored x coordinate
    int mirror_x = t_input_img.m_size.x - 1 - x;
    int idx_dst = y * t_output_img.m_size.x + mirror_x; //y
    
 
    t_output_img.m_p_uchar3[idx_dst] = t_input_img.m_p_uchar3[idx_src];
}

//3 chess overlay
__global__ void kernel_chess_overlay(CudaImg t_input1_img, CudaImg t_input2_img, CudaImg t_output_img, int tile_size, uchar3 highlight_color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= t_input1_img.m_size.x || y >= t_input1_img.m_size.y)
        return;

    int idx = y * t_output_img.m_size.x + x;
    
    //check if the pixel is on a tile border
    bool is_border = (x % tile_size == 0) || (y % tile_size == 0);
    
    if (is_border) {
        t_output_img.m_p_uchar3[idx] = highlight_color;
    }
    else {
        //calculate block coordinates and determine parity
        int block_x = x / tile_size;
        int block_y = y / tile_size;
        int parity = (block_x + block_y) % 2;
        
        //choose source image based on block parity
        if (parity == 0) {
            t_output_img.m_p_uchar3[idx] = t_input1_img.m_p_uchar3[idx];
        }
        else {
            t_output_img.m_p_uchar3[idx] = t_input2_img.m_p_uchar3[idx];
        }
    }
}

//invert image
extern "C" void cu_invert_image(CudaImg t_bgr_cuda_img, bool horizontal)
{
    cudaError_t l_cerr;
    
    // Grid creation
    int l_block_size = 16;
    dim3 l_blocks((t_bgr_cuda_img.m_size.x + l_block_size - 1) / l_block_size, 
                 (t_bgr_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    
    if (horizontal) {
        kernel_invert_horizontal<<<l_blocks, l_threads>>>(t_bgr_cuda_img);
    } else {
        kernel_invert_vertical<<<l_blocks, l_threads>>>(t_bgr_cuda_img);
    }
    
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
        
    cudaDeviceSynchronize();
}

//color removal
extern "C" void cu_remove_color_amount(CudaImg t_bgr_cuda_img, unsigned char r, unsigned char g, unsigned char b, float amount)
{
    cudaError_t l_cerr;
    
    // Grid creation
    int l_block_size = 16;
    dim3 l_blocks((t_bgr_cuda_img.m_size.x + l_block_size - 1) / l_block_size, 
                 (t_bgr_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    
    kernel_remove_color_amount<<<l_blocks, l_threads>>>(t_bgr_cuda_img, r, g, b, amount);
    
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
        
    cudaDeviceSynchronize();
}


// 1 color filter implementation
extern "C" void cu_color_filter(CudaImg t_input_img, CudaImg t_output_img, int color_channel)
{
    cudaError_t l_cerr;
    
    int l_block_size = 16;
    dim3 l_blocks((t_input_img.m_size.x + l_block_size - 1) / l_block_size, 
                 (t_input_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    
    kernel_color_filter<<<l_blocks, l_threads>>>(t_input_img, t_output_img, color_channel);
    
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
        
    cudaDeviceSynchronize();
}

//2 horizontal mirror implementation
extern "C" void cu_mirror_horizontal(CudaImg t_input_img, CudaImg t_output_img)
{
    cudaError_t l_cerr;
    int l_block_size = 16;
    dim3 l_blocks((t_input_img.m_size.x + l_block_size - 1) / l_block_size, 
                 (t_input_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    
    kernel_mirror_horizontal_full<<<l_blocks, l_threads>>>(t_input_img, t_output_img);
    
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
        
    cudaDeviceSynchronize();
}

//3 chess overlay implementation
extern "C" void cu_chess_overlay(CudaImg t_input1_img, CudaImg t_input2_img, CudaImg t_output_img, int tile_size, uchar3 highlight_color)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((t_input1_img.m_size.x + l_block_size - 1) / l_block_size, 
                 (t_input1_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    
    kernel_chess_overlay<<<l_blocks, l_threads>>>(t_input1_img, t_input2_img, t_output_img, tile_size, highlight_color);
    
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
        
    cudaDeviceSynchronize();
}
