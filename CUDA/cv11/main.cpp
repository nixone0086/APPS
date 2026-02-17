#include <stdio.h>
#include <math.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "uni_mem_allocator.h"
#include "cuda_img.h"


void cu_bilinear_scale(CudaImg &input_img, CudaImg &output_img);
void cu_rotate(CudaImg &input_img, CudaImg &output_img, float angle);
void cu_motion_blur(CudaImg &input_img, CudaImg &output_img, float blur_level);
void cu_alpha_blend_insert(CudaImg &background_img, CudaImg &object_img, int x_offset, int y_offset);

// Расчет размера после поворота (по диагонали)
void calculateRotatedSize(int orig_width, int orig_height, int &new_width, int &new_height) {
    float diagonal = sqrt(orig_width * orig_width + orig_height * orig_height);
    new_width = new_height = ceil(diagonal);
}

int main(int argc, char **argv)
{
    // Инициализация унифицированного аллокатора
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    // Загрузка фонового изображения
    cv::Mat background_cv_img = cv::imread("background.jpg", cv::IMREAD_COLOR);
    if (!background_cv_img.data)
    {
        printf("Unable to read file 'background.jpg'\n");
        return 1;
    }

    // Загрузка изображения колеса с альфа-каналом
    cv::Mat wheel_cv_img = cv::imread("wheel.png", cv::IMREAD_UNCHANGED);
    if (!wheel_cv_img.data)
    {
        printf("Unable to read file 'wheel.png'\n");
        return 1;
    }

    // Преобразование в BGRA при необходимости
    cv::Mat wheel_rgba_cv_img;
    if (wheel_cv_img.channels() == 3) {
        cv::cvtColor(wheel_cv_img, wheel_rgba_cv_img, cv::COLOR_BGR2BGRA);
    } else {
        wheel_rgba_cv_img = wheel_cv_img;
    }

    // Расчет размеров повёрнутого изображения
    int rotated_size_x, rotated_size_y;
    calculateRotatedSize(wheel_rgba_cv_img.cols, wheel_rgba_cv_img.rows, rotated_size_x, rotated_size_y);

    // Создание буферов
    cv::Mat rotated_wheel_cv_img(rotated_size_y, rotated_size_x, CV_8UC4);
    cv::Mat blurred_wheel_cv_img(rotated_size_y, rotated_size_x, CV_8UC4);
    cv::Mat frame_cv_img(background_cv_img.size(), CV_8UC3);

    // CUDA структуры
    CudaImg background_cuda_img, wheel_cuda_img, rotated_wheel_cuda_img, blurred_wheel_cuda_img, frame_cuda_img;

    // Инициализация
    background_cuda_img.m_size.x = background_cv_img.cols;
    background_cuda_img.m_size.y = background_cv_img.rows;
    background_cuda_img.m_p_uchar3 = (uchar3 *)background_cv_img.data;

    wheel_cuda_img.m_size.x = wheel_rgba_cv_img.cols;
    wheel_cuda_img.m_size.y = wheel_rgba_cv_img.rows;
    wheel_cuda_img.m_p_uchar4 = (uchar4 *)wheel_rgba_cv_img.data;

    rotated_wheel_cuda_img.m_size.x = rotated_wheel_cv_img.cols;
    rotated_wheel_cuda_img.m_size.y = rotated_wheel_cv_img.rows;
    rotated_wheel_cuda_img.m_p_uchar4 = (uchar4 *)rotated_wheel_cv_img.data;

    blurred_wheel_cuda_img.m_size.x = blurred_wheel_cv_img.cols;
    blurred_wheel_cuda_img.m_size.y = blurred_wheel_cv_img.rows;
    blurred_wheel_cuda_img.m_p_uchar4 = (uchar4 *)blurred_wheel_cv_img.data;

    frame_cuda_img.m_size.x = frame_cv_img.cols;
    frame_cuda_img.m_size.y = frame_cv_img.rows;
    frame_cuda_img.m_p_uchar3 = (uchar3 *)frame_cv_img.data;

    // Параметры анимации
    int total_frames = 3000; // 10 секунд при 30 fps
    float max_rotation_speed = 0.30f; 
    float max_blur_level = 1.0f;

    float motion_ratio = 0.7f; // Сколько процентов времени колесо движется
    int motion_frames = total_frames * motion_ratio;

    // Параметры движения
    int start_x = -rotated_size_x;
    int end_x = background_cv_img.cols;
    float total_distance = end_x - start_x;

    // Видеозапись
    cv::VideoWriter video_writer("wheel_animation.mkv",
                                 cv::VideoWriter::fourcc('X', '2', '6', '4'),
                                 30,
                                 cv::Size(background_cv_img.cols, background_cv_img.rows),
                                 true);

    if (!video_writer.isOpened()) {
        printf("Error: Could not open video writer\n");
        return 1;
    }

    float current_angle = 0.0f;
    float current_speed = 0.0f;

    for (int frame = 0; frame < total_frames; frame++) {
        background_cv_img.copyTo(frame_cv_img);

       
        float current_x;

        if (frame < motion_frames) {
            // Сначала колесо едет вправо, потом влево
            float half_motion = motion_frames / 2.0f;
            float movement_phase = (frame < half_motion) ?
                (float)frame / half_motion :
                (float)(motion_frames - frame) / half_motion;

            current_x = start_x + movement_phase * total_distance;
        } else {
            // После движения — колесо остаётся по центру
            current_x = (background_cv_img.cols - rotated_size_x) / 2;
        }

        
        float t = (float)frame / total_frames;
        current_speed = t * max_rotation_speed;

        current_angle += current_speed;
        if (current_angle > 2 * M_PI)
            current_angle -= 2 * M_PI;

        //  Размытие при вращении 
        float blur_level = (current_speed / max_rotation_speed) * max_blur_level;

        cu_rotate(wheel_cuda_img, rotated_wheel_cuda_img, current_angle);

        if (blur_level > 0.01f) {
            cu_motion_blur(rotated_wheel_cuda_img, blurred_wheel_cuda_img, blur_level);
        } else {
            rotated_wheel_cv_img.copyTo(blurred_wheel_cv_img);
        }

        // Центровка по вертикали
        int current_y = (background_cv_img.rows - rotated_wheel_cv_img.rows) / 2;

        cu_alpha_blend_insert(frame_cuda_img, blurred_wheel_cuda_img, current_x, current_y);

        video_writer.write(frame_cv_img);

        if (frame % 30 == 0) {
            cv::imshow("Animation Preview", frame_cv_img);
            cv::waitKey(1);
        }
    }

    video_writer.release();
    cv::destroyAllWindows();

    return 0;
}
