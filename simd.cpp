#include "utils.hpp"
#include "fft/fft.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace std::chrono;

bool areChannelsEqual(const vector<Mat>& v1, const vector<Mat>& v2) {
    if (v1.size() != v2.size()) return false;

    for (int i = 0; i < v1.size(); i++) {
        if (v1[i].size() != v2[i].size() || v1[i].type() != v2[i].type()) {
            cerr << "Channel " << i << ": size/type mismatch\n";
            return false;
        }
        if (cv::norm(v1[i], v2[i], NORM_L2) != 0.0) {
            cerr << "Channel " << i << ": data mismatch\n";
            return false;
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
    if (img.empty()) {
        cout << "Cannot read image\n";
        return -1;
    }
    img.convertTo(img, CV_32F);
    img /= 255.0;

    Mat psf = motionBlurKernel(psf_length, psf_angle);
    float K = 0.01f;

    // ============ split channels ============
    vector<Mat> serial_ch(3), simd_ch(3);
    split(img, serial_ch);
    split(img, simd_ch);

    // ================= SERIAL =================
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        Mat channel = serial_ch[i];
        if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);
        serial_ch[i] = fft_serial::wienerDeblur_myfft(channel, psf, K);
        if (usePowerOf2) serial_ch[i] = serial_ch[i](Rect(0, 0, img.cols, img.rows));
    }
    auto t1 = high_resolution_clock::now();
    double serial_time = getElapsedMs(t0, t1);
    cout << "Deblurring 3 channels took (serial) : " << serial_time << " ms\n";

    // ================= SIMD AVX2 =================
    auto t2 = high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        Mat channel = simd_ch[i];
        if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);
        simd_ch[i] = fft_simd::wienerDeblur_myfft(channel, psf, K);
        if (usePowerOf2) simd_ch[i] = simd_ch[i](Rect(0, 0, img.cols, img.rows));
    }
    auto t3 = high_resolution_clock::now();
    double simd_time = getElapsedMs(t2, t3);
    cout << "Deblurring 3 channels took (SIMD AVX2): " << simd_time << " ms\n";

    // ========== Compare Output ==========
    if (areChannelsEqual(serial_ch, simd_ch)) {
        cout << "[Success] SIMD AVX2 output is identical to serial.\n";
        printf("[Speedup] %.2fx faster\n", serial_time / simd_time);
    } else {
        cout << "[Error] SIMD AVX2 result differs from serial.\n";
    }

    // ======= merge + white balance + display =======
    Mat merged_float;
    merge(simd_ch, merged_float);

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
