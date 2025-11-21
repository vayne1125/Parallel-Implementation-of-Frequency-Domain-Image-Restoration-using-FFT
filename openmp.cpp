#include "utils.hpp"
#include "fft/fft.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

bool areChannelsEqual(const std::vector<cv::Mat>& vec1, const std::vector<cv::Mat>& vec2, double epsilon = 1e-3) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Channel count mismatch (" << vec1.size() << " vs " << vec2.size() << ").\n";
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        const cv::Mat& mat1 = vec1[i];
        const cv::Mat& mat2 = vec2[i];

        if (mat1.size() != mat2.size() || mat1.type() != mat2.type()) {
            std::cerr << "Error: Size or type mismatch in channel " << i << ".\n";
            return false;
        }

        double diff = cv::norm(mat1, mat2, cv::NORM_INF);
        
        if (diff > epsilon) {
            std::cerr << "[Error] Content mismatch in channel " << i << ".\n";
            std::cerr << "        Max pixel difference: " << diff << " (Threshold: " << epsilon << ")\n";
            return false;
        }
    }
    return true; 
}

int main(int argc, char** argv) {
    if (argc != 5) {
        cout << "Usage: ./openmp <img-path> <psf-length> <psf-angle> <num-threads>\n";
        return -1;
    }
    string img_path = argv[1];
    int psf_length = atoi(argv[2]);
    double psf_angle = atof(argv[3]);
    int num_threads = atoi(argv[4]);
    // fft_openmp::num_threads = num_threads;

    // int total_threads = num_threads;
    // int outer_threads = min(3,num_threads);
    // int inner_threads = total_threads / outer_threads;
    // if (inner_threads < 1) inner_threads = 1;

    cout << "[INFO] " << num_threads << " threads will be used for OpenMP FFT.\n";
    if (num_threads <= 0) {
        cerr << "Error: Number of threads must be greater than 0.\n";
        return -1;
    }

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


    t_start = high_resolution_clock::now();
    // #pragma omp parallel for num_threads(outer_threads)
    for (int i = 0; i < 3; i++) {
        omp_set_num_threads(num_threads);
        Mat channel = channels[i];
        if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);
        channels[i] = fft_openmp::wienerDeblur_myfft(channel, psf, K);
        if (usePowerOf2) channels[i] = channels[i](Rect(0, 0, img.cols, img.rows));
    }
    t_end = high_resolution_clock::now();
    auto openmp_time = getElapsedMs(t_start, t_end);
    cout << "Deblurring 3 channels took(openmp): " << openmp_time << " ms\n";

    if(areChannelsEqual(serial_channels, channels)) {
        cout << "[Success] The results from serial and OpenMP implementations are identical.\n";
        printf("[Speedup] %.2fx ms\n", serial_time/openmp_time);
    } else {
        cout << "[Error] The results from serial and OpenMP implementations differ.\n";
    }

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
