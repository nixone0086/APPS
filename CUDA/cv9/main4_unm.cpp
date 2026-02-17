#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}
void cu_run_grayscale(CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img);
void cu_invert_image(CudaImg t_bgr_cuda_img, bool horizontal);
void cu_remove_color_amount(CudaImg t_bgr_cuda_img, uchar r, uchar g, uchar b, float amount);
void cu_color_filter(CudaImg t_input_img, CudaImg t_output_img, int color_channel);
void cu_mirror_horizontal(CudaImg t_input_img, CudaImg t_output_img);
void cu_chess_overlay(CudaImg t_input1_img, CudaImg t_input2_img, CudaImg t_output_img, int tile_size, uchar3 highlight_color);

int main(int t_numarg, char **t_arg)
{
    //uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (t_numarg < 2)
    {
        printf("Enter picture filename!\n");
        return 1;
    }

    //load image
    cv::Mat l_bgr_cv_img = cv::imread(t_arg[1], cv::IMREAD_COLOR);
    if (!l_bgr_cv_img.data)
    {
        printf("Unable to read file '%s'\n", t_arg[1]);
        return 1;
    }

    //create empty BW image
    cv::Mat l_bw_cv_img(l_bgr_cv_img.size(), CV_8U);

    // data for CUDA
    CudaImg l_bgr_cuda_img, l_bw_cuda_img;
    l_bgr_cuda_img.m_size.x = l_bw_cuda_img.m_size.x = l_bgr_cv_img.size().width;
    l_bgr_cuda_img.m_size.y = l_bw_cuda_img.m_size.y = l_bgr_cv_img.size().height;
    l_bgr_cuda_img.m_p_uchar3 = (uchar3 *)l_bgr_cv_img.data;
    l_bw_cuda_img.m_p_uchar1 = (uchar1 *)l_bw_cv_img.data;

    cu_run_grayscale(l_bgr_cuda_img, l_bw_cuda_img);

    cv::imshow("Original Color", l_bgr_cv_img);
    cv::imshow("GrayScale", l_bw_cv_img);
    cv::waitKey(0);

    //1 Color Filtering
    std::vector<std::string> channel_names = {"Blue", "Green", "Red"};
    std::vector<cv::Mat> filtered_images;
    std::vector<CudaImg> filtered_cuda_imgs;

    // Create filtered images for each channel (R, G, B)
    for (int i = 0; i < 3; i++) {
        cv::Mat filtered_img(l_bgr_cv_img.size(), CV_8UC3);
        filtered_images.push_back(filtered_img);
        
        // Setup CUDA image structure
        CudaImg filtered_cuda_img;
        filtered_cuda_img.m_size.x = filtered_img.size().width;
        filtered_cuda_img.m_size.y = filtered_img.size().height;
        filtered_cuda_img.m_p_uchar3 = (uchar3 *)filtered_img.data;
        filtered_cuda_imgs.push_back(filtered_cuda_img);
        
        cu_color_filter(l_bgr_cuda_img, filtered_cuda_img, i);

        cv::imshow("Color Filter - " + channel_names[i], filtered_img);
        cv::waitKey(0);
    }

    //2 Horizontal Mirroring
    std::vector<cv::Mat> mirrored_images;
    std::vector<CudaImg> mirrored_cuda_imgs;

    //mirrored versions of each filtered image
    for (int i = 0; i < 3; i++) {
        //output image for mirroring
        cv::Mat mirrored_img(l_bgr_cv_img.size(), CV_8UC3);
        mirrored_images.push_back(mirrored_img);
        
        // Setup CUDA image structure
        CudaImg mirrored_cuda_img;
        mirrored_cuda_img.m_size.x = mirrored_img.size().width;
        mirrored_cuda_img.m_size.y = mirrored_img.size().height;
        mirrored_cuda_img.m_p_uchar3 = (uchar3 *)mirrored_img.data;
        mirrored_cuda_imgs.push_back(mirrored_cuda_img);
        
        cu_mirror_horizontal(filtered_cuda_imgs[i], mirrored_cuda_img);

        cv::imshow("Mirrored - " + channel_names[i], mirrored_img);
        cv::waitKey(0);
    }

    //3 Chess Overlay
    cv::Mat chess_img(l_bgr_cv_img.size(), CV_8UC3);
    CudaImg chess_cuda_img;
    chess_cuda_img.m_size.x = chess_img.size().width;
    chess_cuda_img.m_size.y = chess_img.size().height;
    chess_cuda_img.m_p_uchar3 = (uchar3 *)chess_img.data;
    
    //define highlight color (white)
    uchar3 highlight_color;
    highlight_color.x = 255; // Blue
    highlight_color.y = 255; // Green
    highlight_color.z = 255; // Red
    
    cu_chess_overlay(filtered_cuda_imgs[0], l_bgr_cuda_img, chess_cuda_img, 32, highlight_color);
    
    cv::imshow("Chess Overlay", chess_img);
    cv::waitKey(0);

    
    cu_chess_overlay(filtered_cuda_imgs[1], l_bgr_cuda_img, chess_cuda_img, 32, highlight_color);
    
    cv::imshow("Chess Overlay", chess_img);
    cv::waitKey(0);
    
    cu_chess_overlay(filtered_cuda_imgs[2], l_bgr_cuda_img, chess_cuda_img, 32, highlight_color);
    
    cv::imshow("Chess Overlay", chess_img);
    cv::waitKey(0);


    //copy original image for the invert tests
    cv::Mat horizontal_invert_img = l_bgr_cv_img.clone();
    CudaImg horizontal_invert_cuda_img;
    horizontal_invert_cuda_img.m_size.x = horizontal_invert_img.size().width;
    horizontal_invert_cuda_img.m_size.y = horizontal_invert_img.size().height;
    horizontal_invert_cuda_img.m_p_uchar3 = (uchar3*)horizontal_invert_img.data;
    
    //horizontal invert
    cu_invert_image(horizontal_invert_cuda_img, true);
    cv::imshow("Horizontal Invert (Original Method)", horizontal_invert_img);
    cv::waitKey(0);
    
    //vertical invert
    cv::Mat vertical_invert_img = l_bgr_cv_img.clone();
    CudaImg vertical_invert_cuda_img;
    vertical_invert_cuda_img.m_size.x = vertical_invert_img.size().width;
    vertical_invert_cuda_img.m_size.y = vertical_invert_img.size().height;
    vertical_invert_cuda_img.m_p_uchar3 = (uchar3*)vertical_invert_img.data;
    
    cu_invert_image(vertical_invert_cuda_img, false);
    cv::imshow("Vertical Invert (Original Method)", vertical_invert_img);
    cv::waitKey(0);
    
    //color removal
    cv::Mat color_removal_img = l_bgr_cv_img.clone();
    CudaImg color_removal_cuda_img;
    color_removal_cuda_img.m_size.x = color_removal_img.size().width;
    color_removal_cuda_img.m_size.y = color_removal_img.size().height;
    color_removal_cuda_img.m_p_uchar3 = (uchar3*)color_removal_img.data;
    
    cu_remove_color_amount(color_removal_cuda_img, 0, 255, 0, 1.0f);
    cv::imshow("Color Removal", color_removal_img);
    cv::waitKey(0);

    return 0;
}
