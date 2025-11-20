#include "fft.hpp"
#include "../utils.hpp"
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
#include <immintrin.h> // 引入 AVX/AVX2 指令集

using namespace cv;
using namespace std;

namespace fft_simd {

// 對 SoA 資料執行 bit reversal
void bit_reverse_soa(vector<float>& real, vector<float>& imag) {
    int n = (int)real.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            swap(real[i], real[j]);
            swap(imag[i], imag[j]);
        }
    }
}


void fft_radix2_inplace(vector<float>& real, vector<float>& imag, bool inverse)
{
    int n = (int)real.size();
    if (n <= 1) return;

    // 1. Bit reversal
    bit_reverse_soa(real, imag);

    const float sign = inverse ? 1.0f : -1.0f;

    // 2. Butterflies
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2.0f * CV_PI / len * sign;
        complex<float> wlen(cos(ang), sin(ang));
        const float wlen_re = wlen.real();
        const float wlen_im = wlen.imag();
        int half_len = len / 2;

        for (int i = 0; i < n; i += len) {
            
            if (half_len == 1) {//case 1：len = 2（basic butterfly）
                /*
                 X0 = u+v
                 X1 = u−v
                */
                float u_re = real[i];
                float u_im = imag[i];
                float v_re = real[i + 1];
                float v_im = imag[i + 1];
                real[i] = u_re + v_re;
                imag[i] = u_im + v_im;
                real[i + 1] = u_re - v_re;
                imag[i + 1] = u_im - v_im;
                continue; 
            }
            
            if (half_len == 2) {//case 2：len = 4
                // k=0, w=(1,0)
                float u_re = real[i];
                float u_im = imag[i];
                float v_re = real[i + 2];
                float v_im = imag[i + 2];
                real[i] = u_re + v_re;
                imag[i] = u_im + v_im;
                real[i + 2] = u_re - v_re;
                imag[i + 2] = u_im - v_im;

                // k=1, w = wlen
                u_re = real[i + 1];
                u_im = imag[i + 1];
                v_re = real[i + 3];
                v_im = imag[i + 3];
                float v_new_re = -v_im * wlen_im;
                float v_new_im =  v_re * wlen_im;
                real[i + 1] = u_re + v_new_re;
                imag[i + 1] = u_im + v_new_im;
                real[i + 3] = u_re - v_new_re;
                imag[i + 3] = u_im - v_new_im;
                continue; 
            }

            if (half_len == 4) {//case 3：len = 8
                complex<float> w(1.0f, 0.0f);
                for (int k = 0; k < 4; ++k) {
                    float w_re = w.real();
                    float w_im = w.imag();
                    int idx_k = i + k;
                    int idx_k_half = idx_k + 4;

                    float u_re = real[idx_k];
                    float u_im = imag[idx_k];
                    float v_re = real[idx_k_half];
                    float v_im = imag[idx_k_half];

                    float v_new_re = v_re * w_re - v_im * w_im;
                    float v_new_im = v_re * w_im + v_im * w_re;

                    real[idx_k] = u_re + v_new_re;
                    imag[idx_k] = u_im + v_new_im;
                    real[idx_k_half] = u_re - v_new_re;
                    imag[idx_k_half] = u_im - v_new_im;
                    w *= wlen;
                }
                continue;
            }

            //case 4：len ≥ 16 → AVX2 SIMD ×8
            complex<float> w(1.0f, 0.0f);
            alignas(32) float tw_re[8];
            alignas(32) float tw_im[8];

            for (int k = 0; k < half_len; k += 8) {
                complex<float> w_k = w;
                tw_re[0] = w_k.real(); tw_im[0] = w_k.imag(); w_k *= wlen;
                tw_re[1] = w_k.real(); tw_im[1] = w_k.imag(); w_k *= wlen;
                tw_re[2] = w_k.real(); tw_im[2] = w_k.imag(); w_k *= wlen;
                tw_re[3] = w_k.real(); tw_im[3] = w_k.imag(); w_k *= wlen;
                tw_re[4] = w_k.real(); tw_im[4] = w_k.imag(); w_k *= wlen;
                tw_re[5] = w_k.real(); tw_im[5] = w_k.imag(); w_k *= wlen;
                tw_re[6] = w_k.real(); tw_im[6] = w_k.imag(); w_k *= wlen;
                tw_re[7] = w_k.real(); tw_im[7] = w_k.imag();
                w = w_k * wlen; 

                __m256 w_re_v = _mm256_load_ps(tw_re);
                __m256 w_im_v = _mm256_load_ps(tw_im);

                __m256 u_re = _mm256_loadu_ps(&real[i + k]);
                __m256 u_im = _mm256_loadu_ps(&imag[i + k]);
                __m256 v_re = _mm256_loadu_ps(&real[i + k + half_len]);
                __m256 v_im = _mm256_loadu_ps(&imag[i + k + half_len]);

                __m256 v_new_re = _mm256_fmsub_ps(v_re, w_re_v, _mm256_mul_ps(v_im, w_im_v));
                __m256 v_new_im = _mm256_fmadd_ps(v_re, w_im_v, _mm256_mul_ps(v_im, w_re_v));
                
                __m256 out_re_add = _mm256_add_ps(u_re, v_new_re);
                __m256 out_im_add = _mm256_add_ps(u_im, v_new_im);
                __m256 out_re_sub = _mm256_sub_ps(u_re, v_new_re);
                __m256 out_im_sub = _mm256_sub_ps(u_im, v_new_im);

                _mm256_storeu_ps(&real[i + k], out_re_add);
                _mm256_storeu_ps(&imag[i + k], out_im_add);
                _mm256_storeu_ps(&real[i + k + half_len], out_re_sub);
                _mm256_storeu_ps(&imag[i + k + half_len], out_im_sub);
            }
        }
    }
}

// O(n^2) 備用方案
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

// 把 OpenCV 的影像資料 (AoS) 轉換成 FFT 演算法需要的格式(SoA)
void transform_row_inplace(Vec2f* rowPtr, int N, bool inverse)
{
    bool p_of_2 = isPowerOfTwo(N);

    if (p_of_2) {//FFT 只在長度是 2 次方時最快
        //實部與虛部分開，運算速度最快
        vector<float> real(N);
        vector<float> imag(N);

        float* pSrc = (float*)rowPtr;
        float* pDstReal = real.data();
        float* pDstImag = imag.data();

        // 複製，每次處理 8 個像素 (16 個 float)
        // R0 I0 R1 I1 R2 I2 R3 I3 R4 I4 R5 I5 R6 I6 R7 I7 -> R0..R7 and I0..I7
        for (int i = 0; i < N; i += 8) {
            float* p = pSrc + i * 2; 
            __m256 v0 = _mm256_loadu_ps(p + 0);  // [R0 I0 R1 I1]
            __m256 v1 = _mm256_loadu_ps(p + 4);  // [R2 I2 R3 I3]
            __m256 v2 = _mm256_loadu_ps(p + 8);  // [R4 I4 R5 I5]
            __m256 v3 = _mm256_loadu_ps(p + 12); // [R6 I6 R7 I7]

            // 取 real：index = 0,2
            __m256 re01 = _mm256_shuffle_ps(v0, v1, 0x88);//對v0、v1挑出四個元素作重排，最終會挑到[R0 R1 R2 R3]
            __m256 re23 = _mm256_shuffle_ps(v2, v3, 0x88);//[R4 R5 R6 R7]
            // 取 imag：index = 1,3
            __m256 im01 = _mm256_shuffle_ps(v0, v1, 0xDD);//[I0 I1 I2 I3]
            __m256 im23 = _mm256_shuffle_ps(v2, v3, 0xDD);//[I4 I5 I6 I7]

            // 合併
            __m256 re = _mm256_permute2f128_ps(re01, re23, 0x20);//[R0..R7]
            __m256 im = _mm256_permute2f128_ps(im01, im23, 0x20);//[I0..I7]

            _mm256_storeu_ps(pDstReal + i, re);
            _mm256_storeu_ps(pDstImag + i, im);
        }

        //計算
        fft_radix2_inplace(real, imag, inverse);

        // 寫回
        for (int i = 0; i < N; i += 8) {
            __m256 re = _mm256_loadu_ps(pDstReal + i);
            __m256 im = _mm256_loadu_ps(pDstImag + i);

            //照順序的改回交錯
            __m256 v0 = _mm256_unpacklo_ps(re, im); // [R0 I0 R1 I1 R4 I4 R5 I5]
            __m256 v1 = _mm256_unpackhi_ps(re, im); // [R2 I2 R3 I3 R6 I6 R7 I7]
            
            __m256 out0 = _mm256_permute2f128_ps(v0, v1, 0x20);
            __m256 out1 = _mm256_permute2f128_ps(v0, v1, 0x31);

            float* p = pSrc + i * 2;
            _mm256_storeu_ps(p, out0);
            _mm256_storeu_ps(p + 8, out1);
        }
    } else {//非二的冪次使用serial
        vector<complex<float>> buf;
        buf.reserve(N);
        for (int x = 0; x < N; ++x) {
            Vec2f v = rowPtr[x];
            buf.emplace_back(v[0], v[1]);
        }
        
        dft_naive_inplace(buf, inverse);

        for (int x = 0; x < N; ++x) {
            rowPtr[x][0] = buf[x].real();
            rowPtr[x][1] = buf[x].imag();
        }
    }
}

// 2D 轉換
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


// Wiener Deblurring 函數
Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K)
{
    // --- padding ---
    Mat padded;
    int optRows = getOptimalDFTSize(img.rows);
    int optCols = getOptimalDFTSize(img.cols);
    copyMakeBorder(img, padded, 0, optRows - img.rows, 0, optCols - img.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    // --- convert to complex for FFT ---
    Mat planesI[] = { padded.clone(), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planesI, 2, complexI);

    // forward FFT
    my_dft2D_forward(complexI);


    // --- PSF ---
    Mat psfP;
    copyMakeBorder(psf, psfP, 0, optRows - psf.rows, 0, optCols - psf.cols,
                   BORDER_CONSTANT, Scalar::all(0));
    Mat planesH[] = { psfP.clone(), Mat::zeros(psfP.size(), CV_32F) };
    Mat psfComplex;
    merge(planesH, 2, psfComplex);

    my_dft2D_forward(psfComplex);

    const int total = optRows * optCols;

    // 取得平面指標
    Mat GI[2], HI[2];
    split(complexI, GI);
    split(psfComplex, HI);

    float* G_re = (float*)GI[0].data;
    float* G_im = (float*)GI[1].data;
    float* H_re = (float*)HI[0].data;
    float* H_im = (float*)HI[1].data;

    // output
    Mat outRe(optRows, optCols, CV_32F);
    Mat outIm(optRows, optCols, CV_32F);
    float* O_re = (float*)outRe.data;
    float* O_im = (float*)outIm.data;

    __m256 vK = _mm256_set1_ps(K);

    int i = 0;
    for (; i + 7 < total; i += 8)
    {
        __m256 gr = _mm256_loadu_ps(G_re + i);
        __m256 gi = _mm256_loadu_ps(G_im + i);
        __m256 hr = _mm256_loadu_ps(H_re + i);
        __m256 hi = _mm256_loadu_ps(H_im + i);

        // denom = |H|^2 + K = hr² + hi² + K
        __m256 mag2 = _mm256_fmadd_ps(hr, hr, _mm256_mul_ps(hi, hi));
        __m256 denom = _mm256_add_ps(mag2, vK);

        // conj(H) = (hr, -hi)
        __m256 neg_hi = _mm256_sub_ps(_mm256_setzero_ps(), hi);

        // numerator real = gr*hr + gi*hi
        __m256 num_r = _mm256_fmadd_ps(gi, hi, _mm256_mul_ps(gr, hr));

        // numerator imag = -gr*hi + gi*hr
        __m256 num_i = _mm256_fmadd_ps(gi, hr, _mm256_mul_ps(gr, neg_hi));

        // divide
        __m256 invd = _mm256_div_ps(_mm256_set1_ps(1.0f), denom);
        __m256 r = _mm256_mul_ps(num_r, invd);
        __m256 s = _mm256_mul_ps(num_i, invd);

        _mm256_storeu_ps(O_re + i, r);
        _mm256_storeu_ps(O_im + i, s);
    }

    // tail (scalar)
    for (; i < total; ++i)
    {
        float gr = G_re[i], gi = G_im[i];
        float hr = H_re[i], hi = H_im[i];

        float denom = hr*hr + hi*hi + K;
        float nr = gr*hr + gi*hi;
        float ni = -gr*hi + gi*hr;

        O_re[i] = nr / denom;
        O_im[i] = ni / denom;
    }

    // pack 回 complex
    Mat result;
    Mat mergePlanes[] = { outRe, outIm };
    merge(mergePlanes, 2, result);

    // inverse FFT
    my_dft2D_inverse(result);

    // take real part
    Mat restoredPlanes[2];
    split(result, restoredPlanes);
    Mat restored = restoredPlanes[0];

    Mat finalRestored = restored(Rect(0, 0, img.cols, img.rows)).clone();
    normalize(finalRestored, finalRestored, 0, 1, NORM_MINMAX);

    return finalRestored;
}

} // namespace fft_simd
