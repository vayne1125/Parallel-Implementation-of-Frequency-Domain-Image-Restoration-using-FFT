#include "utils.hpp"
#include "fft/fft.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// 比較兩個 Mat 向量是否相等的輔助函數
bool areChannelsEqual(const std::vector<cv::Mat>& vec1, const std::vector<cv::Mat>& vec2) {
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
        
        double diff = cv::norm(mat1, mat2, cv::NORM_L2);
        if (diff != 0.0) {
            // 允許微小的誤差 (因為 AVX FMA 精度通常比純量高，導致結果不完全 bit-exact)
            if (diff > 1.0) { 
                std::cerr << "Error: Content mismatch in channel " << i << " (Diff: " << diff << ").\n";
                return false;
            }
        }
    }
    return true; 
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "Usage: ./fft_image_restoration <img-path> <psf-length> <psf-angle>\n";
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

    // 1. 準備資料
    vector<Mat> serial_channels;
    split(img, serial_channels);

    vector<Mat> simd_channels;
    split(img, simd_channels);

    // 2. 執行 Serial 版本 (作為基準)
    auto t_start = high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        Mat channel = serial_channels[i];
        if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);
        
        // 呼叫 Serial
        serial_channels[i] = fft_serial::wienerDeblur_myfft(channel, psf, K);
        
        if (usePowerOf2) serial_channels[i] = serial_channels[i](Rect(0, 0, img.cols, img.rows));
    }
    auto t_end = high_resolution_clock::now();
    auto serial_time = getElapsedMs(t_start, t_end);
    cout << "Deblurring 3 channels took(serial): " << serial_time << " ms\n";

    // 3. 執行 SIMD 版本
    t_start = high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        Mat channel = simd_channels[i];
        if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);
        
        simd_channels[i] = fft_simd::wienerDeblur_myfft(channel, psf, K); 
        
        if (usePowerOf2) simd_channels[i] = simd_channels[i](Rect(0, 0, img.cols, img.rows));
    }
    t_end = high_resolution_clock::now();
    auto simd_time = getElapsedMs(t_start, t_end);
    cout << "Deblurring 3 channels took(simd): " << simd_time << " ms\n";

    // 4. 驗證結果與輸出加速比
    if(areChannelsEqual(serial_channels, simd_channels)) {
        cout << "[Success] The results from serial and SIMD implementations are identical.\n";
        printf("[Speedup] %.2fx\n", serial_time / simd_time);
    } else {
        cout << "[Warning/Error] The results differ.\n";
        // 註：如果只是微小的浮點數誤差，可能是正常的
    }

    // 5. 後處理與顯示
    Mat merged_float;
    merge(simd_channels, merged_float); // 使用加速後的結果顯示

    Mat merged_Lab, img_orig_Lab;
    cvtColor(merged_float, merged_Lab, COLOR_BGR2Lab);
    cvtColor(img, img_orig_Lab, COLOR_BGR2Lab);

    Mat corrected_Lab = applyWhiteBalance(merged_Lab, img_orig_Lab);

    Mat corrected_BGR;
    cvtColor(corrected_Lab, corrected_BGR, COLOR_Lab2BGR);
    corrected_BGR.convertTo(corrected_BGR, CV_8U, 255.0);

    imshow("Deblurred Color Image (SIMD)", corrected_BGR);
    waitKey(0);
}
