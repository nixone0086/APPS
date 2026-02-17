#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

int main(int t_numarg, char **t_arg)
{
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    cv::Mat background_cv_img = cv::imread("background.jpg", cv::IMREAD_COLOR);
    if (!background_cv_img.data)
    {
        printf("Unable to read background image 'background.jpg'\n");
        return 1;
    }
    cv::Mat object_cv_img = cv::imread("object.png", cv::IMREAD_UNCHANGED);
    if (!object_cv_img.data)
    {
        printf("Unable to read object image 'object.png'\n");
        return 1;
    }
    if (object_cv_img.channels() != 4)
    {
        printf("Object image must have an alpha channel (4 channels)\n");
        return 1;
    }

    CudaImg background_cuda_img, object_cuda_img;

    background_cuda_img.m_size.x = background_cv_img.size().width;
    background_cuda_img.m_size.y = background_cv_img.size().height;
    background_cuda_img.m_p_uchar3 = (uchar3 *)background_cv_img.data;
    
    object_cuda_img.m_size.x = object_cv_img.size().width;
    object_cuda_img.m_size.y = object_cv_img.size().height;
    object_cuda_img.m_p_uchar4 = (uchar4 *)object_cv_img.data;
    

    cv::imshow("Background", background_cv_img);
    cv::imshow("Original", object_cv_img);
    cv::waitKey(0);

    // 1 resize object to half its size using bilinear interpolation
    cv::Mat scaled_object_cv_img(object_cv_img.rows / 2, object_cv_img.cols / 2, CV_8UC4);
    CudaImg scaled_object_cuda_img;
    scaled_object_cuda_img.m_size.x = scaled_object_cv_img.size().width;
    scaled_object_cuda_img.m_size.y = scaled_object_cv_img.size().height;
    scaled_object_cuda_img.m_p_uchar4 = (uchar4 *)scaled_object_cv_img.data;
    
    cu_bilinear_scale(object_cuda_img, scaled_object_cuda_img);
    
    cv::imshow("Scaled Object", scaled_object_cv_img);
    cv::waitKey(0);
    
    // 2 scaled object
    cv::Mat swapped_channels_cv_img(scaled_object_cv_img.size(), CV_8UC4);
    CudaImg swapped_channels_cuda_img;
    swapped_channels_cuda_img.m_size.x = swapped_channels_cv_img.size().width;
    swapped_channels_cuda_img.m_size.y = swapped_channels_cv_img.size().height;
    swapped_channels_cuda_img.m_p_uchar4 = (uchar4 *)swapped_channels_cv_img.data;
    
    cu_swap_channels(scaled_object_cuda_img, swapped_channels_cuda_img);
    
    cv::imshow("Swapped Channels", swapped_channels_cv_img);
    cv::waitKey(0);
    

cv::Mat rotated_cv_img(scaled_object_cv_img.rows, scaled_object_cv_img.cols, CV_8UC4); 
CudaImg rotated_cuda_img;
rotated_cuda_img.m_size.x = rotated_cv_img.size().width;
rotated_cuda_img.m_size.y = rotated_cv_img.size().height;
rotated_cuda_img.m_p_uchar4 = (uchar4 *)rotated_cv_img.data;
cu_rotate90(scaled_object_cuda_img, rotated_cuda_img);

cv::imshow("Rotated 90 Degrees", rotated_cv_img);
cv::waitKey(0);

    cv::Mat grayscale_cv_img(scaled_object_cv_img.size(), CV_8UC4);
    CudaImg grayscale_cuda_img;
    grayscale_cuda_img.m_size.x = grayscale_cv_img.size().width;
    grayscale_cuda_img.m_size.y = grayscale_cv_img.size().height;
    grayscale_cuda_img.m_p_uchar4 = (uchar4 *)grayscale_cv_img.data;
    
    cu_to_grayscale(scaled_object_cuda_img, grayscale_cuda_img);
    
    cv::imshow("Grayscale", grayscale_cv_img);
    cv::waitKey(0);
    
    // 3 create a copy of the background for the final result
    cv::Mat result_cv_img = background_cv_img.clone();
    CudaImg result_cuda_img;
    result_cuda_img.m_size.x = result_cv_img.size().width;
    result_cuda_img.m_size.y = result_cv_img.size().height;
    result_cuda_img.m_p_uchar3 = (uchar3 *)result_cv_img.data;
    
    int padding = 20;
    
    int bg_width = background_cv_img.cols;
    int bg_height = background_cv_img.rows;
    
    cu_alpha_blend_insert(result_cuda_img, scaled_object_cuda_img, 
                          padding, 
                          padding);
    
    cu_alpha_blend_insert(result_cuda_img, swapped_channels_cuda_img, 
                          bg_width - swapped_channels_cuda_img.m_size.x - padding, 
                          padding);
    
    cu_alpha_blend_insert(result_cuda_img, rotated_cuda_img, 
                          padding, 
                          bg_height - rotated_cuda_img.m_size.y - padding);
    
    cu_alpha_blend_insert(result_cuda_img, grayscale_cuda_img, 
                          bg_width - grayscale_cuda_img.m_size.x - padding, 
                          bg_height - grayscale_cuda_img.m_size.y - padding);
    
    // 5
    cv::imshow("Final Result", result_cv_img);
    cv::imwrite("result.png", result_cv_img);
    cv::waitKey(0);

    return 0;
}
