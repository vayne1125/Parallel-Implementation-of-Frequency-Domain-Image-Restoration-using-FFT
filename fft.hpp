#pragma once
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
using namespace cv;
using namespace std;

// FFT / DFT
void fft_radix2_inplace(vector<complex<float>>& a, bool inverse);
void dft_naive_inplace(vector<complex<float>>& a, bool inverse);
void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse);
void my_dft2D(Mat& complexMat, bool inverse);
inline void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
inline void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }

// Wiener deblur using custom FFT
Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K);
