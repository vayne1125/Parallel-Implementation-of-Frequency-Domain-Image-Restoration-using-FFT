#include "utils.hpp"
#include "fft/fft.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

bool areChannelsEqual(const std::vector<cv::Mat>& vec1, const std::vector<cv::Mat>& vec2, double epsilon = 1e-3) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Channel count mismatch.\n";
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        const cv::Mat& mat1 = vec1[i];
        const cv::Mat& mat2 = vec2[i];

        if (mat1.size() != mat2.size() || mat1.type() != mat2.type()) {
            std::cerr << "Error: Size/Type mismatch.\n";
            return false;
        }

        // 1. 計算 Max Diff (L-inf norm)
        double diff = cv::norm(mat1, mat2, cv::NORM_INF);

        // 2. 計算 PSNR (更科學的指標)
        // MSE = sum((a-b)^2) / N
        double mse = cv::norm(mat1, mat2, cv::NORM_L2SQR) / (double)(mat1.total());
        // PSNR = 10 * log10(MAX^2 / MSE), 假設影像範圍是 0~1
        double psnr = (mse > 1e-10) ? (10.0 * log10(1.0 / mse)) : 100.0;

        // 判斷邏輯：
        // 如果誤差真的很小 (< epsilon) -> 通過
        // 或者如果 PSNR 很高 (> 30dB) -> 也算通過 (容許浮點數誤差)
        if (diff > epsilon) {
            if (psnr >= 30.0) {
                cout << "[Info] Channel " << i << " has floating point drift."
                     << "\n       Max Diff: " << diff 
                     << "\n       PSNR: " << psnr << " dB (Excellent! > 30dB is good)"
                     << "\n       -> Verification PASSED (Relaxed)." << endl;
            } else {
                std::cerr << "[Error] Content mismatch in channel " << i << ".\n";
                std::cerr << "       Max pixel difference: " << diff << " (Threshold: " << epsilon << ")\n";
                std::cerr << "       PSNR: " << psnr << " dB (Too low!)\n";
                return false;
            }
        }
    }
    return true; 
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "Usage: ./gpu <img-path> <psf-length> <psf-angle>\n";
        return -1;
    }
    string img_path = argv[1];
    int psf_length = atoi(argv[2]);
    double psf_angle = atof(argv[3]);

    bool usePowerOf2 = true; 

    Mat img = imread(img_path, IMREAD_COLOR);
    if (img.empty()) { cout << "Cannot read image\n"; return -1; }
    img.convertTo(img, CV_32F);
    img /= 255.0;

    Mat psf = motionBlurKernel(psf_length, psf_angle); 
    float K = 0.01f;

    vector<Mat> serial_channels;
    split(img, serial_channels);

    vector<Mat> channels;
    split(img, channels);

    auto t_start = high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        Mat channel = serial_channels[i];
        if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);
        serial_channels[i] = fft_serial::wienerDeblur_myfft(channel, psf, K);
        if (usePowerOf2) serial_channels[i] = serial_channels[i](Rect(0, 0, img.cols, img.rows));
    }
    auto t_end = high_resolution_clock::now();
    auto serial_time = getElapsedMs(t_start, t_end);
    cout << "Deblurring 3 channels took(serial): " << serial_time << " ms\n";
    
    
    cudaFree(0);

    fft_gpu::wienerDeblur_RGB_optimized(channels, psf, K);



    t_start = high_resolution_clock::now();
    fft_gpu::wienerDeblur_RGB_optimized(channels, psf, K);
    t_end = high_resolution_clock::now();
    auto gpu_time = getElapsedMs(t_start, t_end);
    cout << "Deblurring 3 channels took(gpu[optimize]): " << gpu_time << " ms\n";
    printf("[Speedup] %.2fx ms\n", serial_time/gpu_time);


    t_start = high_resolution_clock::now();
    fft_gpu::wienerDeblur_RGB_naive(channels, psf, K);
    t_end = high_resolution_clock::now();
    gpu_time = getElapsedMs(t_start, t_end);
    cout << "Deblurring 3 channels took(gpu): " << gpu_time << " ms\n";
    printf("[Speedup] %.2fx ms\n", serial_time/gpu_time);

    
    // if(areChannelsEqual(serial_channels, channels)) {
    //     cout << "[Success] The results from serial and GPU implementations are identical.\n";
    //     printf("[Speedup] %.2fx ms\n", serial_time/gpu_time);
    // } else {
    //     cout << "[Error] The results from serial and GPU implementations differ.\n";
    // }

    Mat merged_float;
    merge(channels, merged_float);

    Mat merged_Lab, img_orig_Lab;
    cvtColor(merged_float, merged_Lab, COLOR_BGR2Lab);
    cvtColor(img, img_orig_Lab, COLOR_BGR2Lab);

    Mat corrected_Lab = applyWhiteBalance(merged_Lab, img_orig_Lab);

    Mat corrected_BGR;
    cvtColor(corrected_Lab, corrected_BGR, COLOR_Lab2BGR);
    corrected_BGR.convertTo(corrected_BGR, CV_8U, 255.0);

    // imshow("Deblurred Color Image", corrected_BGR);
    waitKey(0);
}
