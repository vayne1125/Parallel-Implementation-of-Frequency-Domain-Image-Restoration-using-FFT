#include "fft.hpp"
#include "../utils.hpp"
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
using namespace cv;
using namespace std;
using namespace std::chrono;

namespace fft_openmp {
// ============================================================
// CPU Timer Helper
// ============================================================
struct CpuTimer {
    string name;
    high_resolution_clock::time_point start;

    CpuTimer(string n) : name(n) {
        start = high_resolution_clock::now();
    }

    ~CpuTimer() {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start).count();
        cout << "[" << name << "] Time: " << duration / 1000.0 << " ms" << endl;
    }
};

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
    // 使用 thread_local 避免重複 malloc
    static thread_local vector<complex<float>> buf;
    
    if (buf.size() != N) {
        buf.resize(N);
    }

    complex<float>* bufPtr = buf.data();
    for (int x = 0; x < N; ++x) {
        bufPtr[x] = {rowPtr[x][0], rowPtr[x][1]};
    }

    if (isPowerOfTwo(N)) fft_radix2_inplace(buf, inverse);
    else dft_naive_inplace(buf, inverse);

    for (int x = 0; x < N; ++x) {
        rowPtr[x][0] = bufPtr[x].real();
        rowPtr[x][1] = bufPtr[x].imag();
    }
}

// Parallel Transpose with Tiling (Block-based) optimization
// src: M x N, dst: N x M
void transpose_parallel(const Mat& src, Mat& dst) {
    int M = src.rows;
    int N = src.cols;

    // Allocate destination matrix (N x M)
    dst.create(N, M, src.type());

    // Block size for cache locality (tuning parameter, usually 16 or 32)
    const int BLOCK_SIZE = 32;

    // Parallelize over row blocks of the SOURCE matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            
            // Handle boundary conditions (if matrix size is not multiple of BLOCK_SIZE)
            int iEnd = std::min(i + BLOCK_SIZE, M);
            int jEnd = std::min(j + BLOCK_SIZE, N);

            // Process the block
            for (int row = i; row < iEnd; ++row) {
                // Pre-calculate pointers for speed
                const Vec2f* srcRowPtr = src.ptr<Vec2f>(row);
                
                for (int col = j; col < jEnd; ++col) {
                    // Transpose: dst(col, row) = src(row, col)
                    // Note: Writing to dst is scattered (strided), reading from src is sequential
                    // inside this small block, cache misses are minimized.
                    dst.ptr<Vec2f>(col)[row] = srcRowPtr[col];
                }
            }
        }
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
    #pragma omp parallel for
    for (int r = 0; r < M; ++r) {
        Vec2f* rowPtr = complexMat.ptr<Vec2f>(r);
        transform_row_inplace(rowPtr, N, inverse);
    }

    // 2) transpose
    Mat t;
    transpose_parallel(complexMat, t);

    // 3) row-wise transform on transposed (each row length M)
    #pragma omp parallel for
    for (int r = 0; r < t.rows; ++r) {
        Vec2f* rowPtr = t.ptr<Vec2f>(r);
        transform_row_inplace(rowPtr, t.cols, inverse);
    }

    // 4) transpose back into original
    transpose_parallel(t, complexMat);
}

void makeComplexPadded(const Mat& src, Mat& dst, int optRows, int optCols) {
    dst = Mat::zeros(optRows, optCols, CV_32FC2);

    int rows = src.rows;
    int cols = src.cols;

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        const float* srcPtr = src.ptr<float>(i);
        Vec2f* dstPtr = dst.ptr<Vec2f>(i);
        for (int j = 0; j < cols; ++j) {
            dstPtr[j][0] = srcPtr[j]; 
            dstPtr[j][1] = 0.0f; 
        }
    }
}

Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K) {
    int optRows = getOptimalDFTSize(img.rows);
    int optCols = getOptimalDFTSize(img.cols);

    // 1. 準備資料 (Padding)
    Mat complexI, psfComplex;
    {
        // CpuTimer t("OpenMP: Prep & Padding");
        // 這裡包含了 makeComplexPadded 的邏輯
        makeComplexPadded(img, complexI, optRows, optCols);

        Mat psfPadded;
        makeComplexPadded(psf, psfComplex, optRows, optCols);
    }

    // 2. Forward FFT (Image)
    {
        // CpuTimer t("OpenMP: FFT Image");
        my_dft2D_forward(complexI);
    }

    // 3. Forward FFT (PSF)
    {
        // CpuTimer t("OpenMP: FFT PSF");
        my_dft2D_forward(psfComplex);
    }

    // 4. Wiener Filter Calculation
    {
        // CpuTimer t("OpenMP: Wiener Filter");
        int rows = complexI.rows;
        int cols = complexI.cols;

        #pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            Vec2f* ptrG = complexI.ptr<Vec2f>(i);
            const Vec2f* ptrH = psfComplex.ptr<Vec2f>(i);

            for (int j = 0; j < cols; ++j) {
                float Gr = ptrG[j][0]; float Gi = ptrG[j][1];
                float Hr = ptrH[j][0]; float Hi = ptrH[j][1];

                float mag2 = Hr * Hr + Hi * Hi;
                float denom = mag2 + K;
                float invDenom = (denom > 1e-8f) ? (1.0f / denom) : 0.0f;

                float numRe = Gr * Hr + Gi * Hi;
                float numIm = Gi * Hr - Gr * Hi;

                ptrG[j][0] = numRe * invDenom;
                ptrG[j][1] = numIm * invDenom;
            }
        }
    }

    // 5. Inverse FFT
    {
        // CpuTimer t("OpenMP: IFFT");
        my_dft2D_inverse(complexI);
    }

    Mat finalRestored;
    // 6. Post-processing
    {
        // CpuTimer t("OpenMP: Post-process");
        finalRestored = Mat(img.rows, img.cols, CV_32F);
        
        #pragma omp parallel for
        for(int i = 0; i < img.rows; ++i) {
            const Vec2f* srcPtr = complexI.ptr<Vec2f>(i);
            float* dstPtr = finalRestored.ptr<float>(i);
            for(int j = 0; j < img.cols; ++j) {
                dstPtr[j] = srcPtr[j][0];
            }
        }
        normalize(finalRestored, finalRestored, 0, 1, NORM_MINMAX);
    }

    return finalRestored;
}
}