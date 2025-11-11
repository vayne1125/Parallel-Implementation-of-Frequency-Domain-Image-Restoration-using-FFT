#include "utils.hpp"
#include <opencv2/opencv.hpp>

double getElapsedMs(high_resolution_clock::time_point start,
                    high_resolution_clock::time_point end) {
    return duration<double, std::milli>(end - start).count();
}

Mat motionBlurKernel(int size, double angle) {
    Mat kernel = Mat::zeros(size, size, CV_32F);
    Point center(size / 2, size / 2);
    for (int i = 0; i < size; i++)
        kernel.at<float>(center.y, i) = 1.0 / size;
    Mat rot = getRotationMatrix2D(center, angle, 1);
    Mat rotated;
    warpAffine(kernel, rotated, rot, kernel.size());
    return rotated;
}

int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

Mat autoPadToPowerOfTwo(const Mat& src) {
    int newRows = nextPowerOfTwo(src.rows);
    int newCols = nextPowerOfTwo(src.cols);
    Mat padded;
    copyMakeBorder(src, padded, 0, newRows - src.rows, 0, newCols - src.cols,
                   BORDER_CONSTANT, Scalar::all(0));
    return padded;
}

bool isPowerOfTwo(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

Mat applyWhiteBalance(const Mat& img_Lab, const Mat& img_orig_Lab) {
    vector<Mat> orig_channels, deblur_channels;
    split(img_orig_Lab, orig_channels);
    split(img_Lab, deblur_channels);

    double avgL_orig = mean(orig_channels[0])[0];
    double avgL_deblur = mean(deblur_channels[0])[0];
    double gain = avgL_orig / (avgL_deblur + 1e-6);

    deblur_channels[0] = deblur_channels[0] * gain;
    cv::min(deblur_channels[0], 100.0f, deblur_channels[0]);
    cv::max(deblur_channels[0], 0.0f, deblur_channels[0]);

    Mat corrected_Lab;
    merge(deblur_channels, corrected_Lab);
    return corrected_Lab;
}
