#include "fft.hpp"
#include "../utils.hpp"
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
using namespace cv;
using namespace std;

namespace fft_openmp {
// iterative radix-2 Cooley-Tukey FFT (in-place)
// a: vector of complex<float>, n must be power-of-two
// inverse: if true compute inverse transform (no scaling)
void fft_radix2_inplace(vector<complex<float>>& a, bool inverse)
{
    int n = (int)a.size();
    if (n <= 1) return;
    // bit reversal permutation
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    // butterflies
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2.0f * CV_PI / len * (inverse ? 1.0f : -1.0f);
        complex<float> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            complex<float> w(1.0f, 0.0f);
            for (int k = 0; k < len/2; ++k) {
                complex<float> u = a[i + k];
                complex<float> v = a[i + k + len/2] * w;
                a[i + k] = u + v;
                a[i + k + len/2] = u - v;
                w *= wlen;
            }
        }
    }
    // Note: do NOT scale here; caller may scale if needed
}

// naive direct DFT (O(n^2)) for arbitrary n
void dft_naive_inplace(vector<complex<float>>& a, bool inverse)
{
    int n = (int)a.size();
    if (n <= 1) return;
    vector<complex<float>> out(n);
    const float sign = inverse ? 1.0f : -1.0f;
    for (int k = 0; k < n; ++k) {
        complex<float> sum(0,0);
        for (int t = 0; t < n; ++t) {
            float ang = 2.0f * CV_PI * k * t / n * sign;
            complex<float> w(cos(ang), sin(ang));
            sum += a[t] * w;
        }
        out[k] = sum;
    }
    a.swap(out);
}

// perform 1D transform on a single row (CV_32FC2) length = N
void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse)
{
    // copy to complex buffer
    vector<complex<float>> buf;
    buf.reserve(N);
    for (int x = 0; x < N; ++x) {
        Vec2f v = rowPtr[x];
        buf.emplace_back(v[0], v[1]);
    }

    if (isPowerOfTwo(N)) fft_radix2_inplace(buf, inverse);
    else dft_naive_inplace(buf, inverse);

    // write back
    for (int x = 0; x < N; ++x) {
        rowPtr[x][0] = buf[x].real();
        rowPtr[x][1] = buf[x].imag();
    }
}

// 2D separable transform: row-wise FFT, transpose, row-wise FFT, transpose back
// works in-place on CV_32FC2 Mat
// inverse: false = forward DFT, true = inverse DFT (no scaling)
void my_dft2D(Mat& complexMat, bool inverse)
{
    CV_Assert(complexMat.type() == CV_32FC2);

    int M = complexMat.rows;
    int N = complexMat.cols;

    // 1) row-wise transform (each row length N)
    for (int r = 0; r < M; ++r) {
        Vec2f* rowPtr = complexMat.ptr<Vec2f>(r);
        transform_row_inplace(rowPtr, N, inverse);
    }

    // 2) transpose
    Mat t;
    transpose(complexMat, t); // now size: N x M

    // 3) row-wise transform on transposed (each row length M)
    for (int r = 0; r < t.rows; ++r) {
        Vec2f* rowPtr = t.ptr<Vec2f>(r);
        transform_row_inplace(rowPtr, t.cols, inverse);
    }

    // 4) transpose back into original
    transpose(t, complexMat);

}

Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K) {
    Mat padded;
    int optRows = getOptimalDFTSize(img.rows);
    int optCols = getOptimalDFTSize(img.cols);
    copyMakeBorder(img, padded, 0, optRows - img.rows, 0, optCols - img.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {padded, Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);            // CV_32FC2
    my_dft2D_forward(complexI);            // <--- 使用自製 FFT (2D)

    // PSF
    Mat psfPadded;
    copyMakeBorder(psf, psfPadded, 0, optRows - psf.rows, 0, optCols - psf.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat psfPlanes[] = {psfPadded, Mat::zeros(psfPadded.size(), CV_32F)};
    Mat psfComplex;
    merge(psfPlanes, 2, psfComplex);
    my_dft2D_forward(psfComplex);          // <--- 使用自製 FFT (2D)

    // denom = |H|^2 + K
    Mat denom;
    // mulSpectrums(psfComplex, psfComplex, denom, 0, true);
    // Implement |H|^2 manually (real^2 + imag^2) in single-channel float
    Mat planesH[2];
    split(psfComplex, planesH);
    Mat mag2;
    magnitude(planesH[0], planesH[1], mag2); // sqrt(re^2+im^2)
    // mag2 currently = sqrt(|H|^2), so square it:
    mag2 = mag2.mul(mag2); // now |H|^2
    denom = mag2 + Scalar::all(K);

    // psf conjugate
    planesH[1] = -planesH[1];
    Mat psfConj;
    merge(planesH, 2, psfConj);

    // numerator = G * H_conj
    Mat numerator;
    // complex multiply: (a+ib)*(c+id) = (ac - bd) + i(ad + bc)
    {
        Mat A[2], B[2], C[2];
        split(complexI, A); split(psfConj, B);
        C[0] = A[0].mul(B[0]) - A[1].mul(B[1]);
        C[1] = A[0].mul(B[1]) + A[1].mul(B[0]);
        merge(C, 2, numerator);
    }

    // divide by denom (real scalar per frequency): result = numerator / denom
    Mat result = Mat::zeros(numerator.size(), numerator.type());
    {
        Mat numPlanes[2], outP[2];
        split(numerator, numPlanes);
        outP[0] = numPlanes[0] / denom;
        outP[1] = numPlanes[1] / denom;
        merge(outP, 2, result);
    }

    // inverse 2D transform
    my_dft2D_inverse(result); // <--- 使用自製 IDFT (2D)
    // NOTE: our inverse does NOT scale; OpenCV idft default also does not scale unless DFT_SCALE used.
    // If you want scaled inverse (so result is real-space amplitude), scale by 1/(optRows*optCols)
    Mat restoredPlanes[2];
    split(result, restoredPlanes);
    Mat restored = restoredPlanes[0];

    // crop to original size
    Mat finalRestored = restored(Rect(0, 0, img.cols, img.rows)).clone();

    // normalize for display (optional)
    normalize(finalRestored, finalRestored, 0, 1, NORM_MINMAX);

    return finalRestored;
}
}