#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// 建立 motion blur kernel
Mat motionBlurKernel(int size, double angle) {
    Mat kernel = Mat::zeros(size, size, CV_32F);
    Point center(size / 2, size / 2);
    for (int i = 0; i < size; i++)
        kernel.at<float>(center.y, i) = 1.0 / size;

    // 旋轉 kernel
    Mat rot = getRotationMatrix2D(center, angle, 1);
    Mat rotated;
    warpAffine(kernel, rotated, rot, kernel.size());
    return rotated;
}

// 維納反卷積函式
Mat wienerDeblur(const Mat& img, const Mat& psf, float K) {
    Mat padded;
    int optRows = getOptimalDFTSize(img.rows);
    int optCols = getOptimalDFTSize(img.cols);
    copyMakeBorder(img, padded, 0, optRows - img.rows, 0, optCols - img.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {padded, Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);

    // 對 PSF 做同樣 FFT
    Mat psfPadded;
    copyMakeBorder(psf, psfPadded, 0, optRows - psf.rows, 0, optCols - psf.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat psfPlanes[] = {psfPadded, Mat::zeros(psfPadded.size(), CV_32F)};
    Mat psfComplex;
    merge(psfPlanes, 2, psfComplex);
    dft(psfComplex, psfComplex);

    // H*(u,v) / (|H|^2 + K)
    Mat denom;
    mulSpectrums(psfComplex, psfComplex, denom, 0, true); // |H|^2
    add(denom, Scalar::all(K), denom);

    Mat psfConj;
    Mat planesH[2];
    split(psfComplex, planesH);
    planesH[1] *= -1; // conjugate
    merge(planesH, 2, psfConj);

    Mat numerator, result;
    mulSpectrums(complexI, psfConj, numerator, 0);
    divSpectrums(numerator, denom, result, 0);

    idft(result, result);
    split(result, planes);
    Mat restored = planes[0](Rect(0, 0, img.cols, img.rows));
    normalize(restored, restored, 0, 1, NORM_MINMAX);
    return restored;
}

int main() {
    Mat img = imread("input/car_blurred.png", IMREAD_COLOR);
    if (img.empty()) {
        cout << "Cannot read image\n";
        return -1;
    }

    img.convertTo(img, CV_32F);
    img /= 255.0; // Normalize to [0,1]

    // 模糊 kernel（根據模糊方向與長度可自行調整）
    Mat psf = motionBlurKernel(40, 45); // 長度=21, 角度=0°
    float K = 0.01f;                   // 維納濾波係數

    // 分離 RGB 三通道
    vector<Mat> channels;
    split(img, channels);

    for (int i = 0; i < 3; i++) {
        channels[i] = wienerDeblur(channels[i], psf, K);
    }

    Mat merged;
    merge(channels, merged);
    merged.convertTo(merged, CV_8U, 255.0);

    imshow("Deblurred Color Image", merged);
    waitKey(0);
}
