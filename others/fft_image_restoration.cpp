#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std::chrono;

using namespace cv;
using namespace std;

// 計算時間的簡單函式
double getElapsedMs(high_resolution_clock::time_point start,
    high_resolution_clock::time_point end) {
        return duration<double, std::milli>(end - start).count();
}

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
// ---------- utility ----------
static inline bool isPowerOfTwo(int n) {
    return n > 0 && ( (n & (n-1)) == 0 );
}

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

// thin wrappers matching semantic names
void my_dft2D_forward(Mat& complexMat) { my_dft2D(complexMat, false); }
void my_dft2D_inverse(Mat& complexMat) { my_dft2D(complexMat, true); }

// ----------------- Example: integrate into Wiener deblur -----------------
// Replace dft(complexI, complexI) with my_dft2D_forward(complexI)
// Replace idft(result, result) with my_dft2D_inverse(result)
// (No automatic scaling applied)

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

// 計算最接近 >= n 的 2 的冪次
int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// 自動 padding 函式
Mat autoPadToPowerOfTwo(const Mat& src) {
    int newRows = nextPowerOfTwo(src.rows);
    int newCols = nextPowerOfTwo(src.cols);
    Mat padded;
    copyMakeBorder(src, padded, 0, newRows - src.rows, 0, newCols - src.cols, BORDER_CONSTANT, Scalar::all(0));
    return padded;
}

// img_Lab: float Lab image
// img_orig_Lab: 原始 float Lab image
// 返回校正後的 Lab image
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

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "Usage: ./fft_image_restoration <img-path> <psf-length> <psf-angle>\n";
        return -1;
    }
    string img_path = argv[1];
    int psf_length = atoi(argv[2]);
    double psf_angle = atof(argv[3]);

    bool usePowerOf2 = true; // 控制是否自動補到 2 的冪次

    Mat img = imread(img_path, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Cannot read image\n";
        return -1;
    }

    img.convertTo(img, CV_32F);
    img /= 255.0; // Normalize to [0,1]

    // 模糊 kernel（根據模糊方向與長度可自行調整）
    Mat psf = motionBlurKernel(psf_length, psf_angle); 
    float K = 0.01f;                   // 維納濾波係數

    // 分離 RGB 三通道
    vector<Mat> channels;
    split(img, channels);

    auto t_start = high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        Mat channel = channels[i];
        if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);

        channels[i] = wienerDeblur_myfft(channel, psf, K);

        // 去掉 padding 部分，只保留原圖範圍
        if (usePowerOf2) channels[i] = channels[i](Rect(0, 0, img.cols, img.rows));
    }
    auto t_end = high_resolution_clock::now();
    cout << "Deblurring 3 channels took: " << getElapsedMs(t_start, t_end) << " ms\n";

    Mat merged_float;
    merge(channels, merged_float); // float, [0,1]

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
