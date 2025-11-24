#pragma once
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
using namespace cv;
using namespace std;

// FFT / DFT
namespace fft_serial {
    void fft_radix2_inplace(vector<complex<float>>& a, bool inverse);
    void dft_naive_inplace(vector<complex<float>>& a, bool inverse);
    void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse);
    void my_dft2D(Mat& complexMat, bool inverse);
    inline void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
    inline void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }
    // Wiener deblur using custom FFT
    Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K);
}

namespace fft_simd {
    void fft_radix2_inplace(vector<complex<float>>& a, bool inverse);
    void dft_naive_inplace(vector<complex<float>>& a, bool inverse);
    void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse);
    void my_dft2D(Mat& complexMat, bool inverse);
    inline void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
    inline void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }
    // Wiener deblur using custom FFT
    Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K);
}

namespace fft_gpu {
    void wienerDeblur_RGB_naive(vector<Mat>& channels, const Mat& psf, float K);
    void wienerDeblur_RGB_optimized(vector<Mat>& channels, const Mat& psf, float K);
    // void fft_radix2_inplace(vector<complex<float>>& a, bool inverse);
    void fft_radix2_kernel(float* data, int n, bool inverse);
    // void dft_naive_inplace(vector<complex<float>>& a, bool inverse);
    void dft_naive_kernel(float* data, int n, bool inverse);
    // void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse);
    void transform_row_kernel(float* rowPtr, int N, bool inverse);
    void my_dft2D(Mat& complexMat, bool inverse);
    inline void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
    inline void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }
    // Wiener deblur using custom FFT
    Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K);
}

namespace fft_openmp {
    // inline int num_threads; // Set number of threads
    void fft_radix2_inplace(vector<complex<float>>& a, bool inverse);
    void dft_naive_inplace(vector<complex<float>>& a, bool inverse);
    void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse);
    void my_dft2D(Mat& complexMat, bool inverse);
    inline void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
    inline void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }
    void transpose_parallel(const Mat& src, Mat& dst);
    // Wiener deblur using custom FFT
    Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K);
}


namespace fft_mpi {
    void fft_radix2_inplace(vector<complex<float>>& a, bool inverse);
    void dft_naive_inplace(vector<complex<float>>& a, bool inverse);
    void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse);
    void my_dft2D(Mat& complexMat, bool inverse);
    inline void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
    inline void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }
    // Wiener deblur using custom FFT
    Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K);
}

namespace fft_parallel {
    void fft_radix2_inplace(vector<complex<float>>& a, bool inverse);
    void dft_naive_inplace(vector<complex<float>>& a, bool inverse);
    void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse);
    void my_dft2D(Mat& complexMat, bool inverse);
    inline void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
    inline void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }
    // Wiener deblur using custom FFT
    Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K);
}

// namespace fft_parallel_name {
//     void fft_radix2_inplace(vector<complex<float>>& a, bool inverse);
//     void dft_naive_inplace(vector<complex<float>>& a, bool inverse);
//     void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse);
//     void my_dft2D(Mat& complexMat, bool inverse);
//     inline void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
//     inline void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }
//     // Wiener deblur using custom FFT
//     Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K);
// }

