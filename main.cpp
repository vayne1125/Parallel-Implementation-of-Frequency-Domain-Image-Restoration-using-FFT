#include "utils.hpp"
#include "fft/fft.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace std::chrono;

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

    vector<Mat> channels;
    split(img, channels);

    auto t_start = high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        Mat channel = channels[i];
        if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);

        // todo: change namespace to fft_simd or others to test different implementations
        channels[i] = fft_simd::wienerDeblur_myfft(channel, psf, K);

        if (usePowerOf2) channels[i] = channels[i](Rect(0, 0, img.cols, img.rows));
    }
    auto t_end = high_resolution_clock::now();
    cout << "Deblurring 3 channels took: " << getElapsedMs(t_start, t_end) << " ms\n";

    Mat merged_float;
    merge(channels, merged_float);

    Mat merged_Lab, img_orig_Lab;
    cvtColor(merged_float, merged_Lab, COLOR_BGR2Lab);
    cvtColor(img, img_orig_Lab, COLOR_BGR2Lab);

    Mat corrected_Lab = applyWhiteBalance(merged_Lab, img_orig_Lab);

    Mat corrected_BGR;
    cvtColor(corrected_Lab, corrected_BGR, COLOR_Lab2BGR);
    corrected_BGR.convertTo(corrected_BGR, CV_8U, 255.0);

    imshow("Deblurred Color Image", corrected_BGR);
    waitKey(0);
}
