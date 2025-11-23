#include "fft.hpp"
#include "../utils.hpp"
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
#include <mpi.h>
#include <immintrin.h> // 引入 AVX/AVX2 指令集

using namespace cv;
using namespace std;

namespace fft_mpi_simd {

static int callCount = 0;
static const int CHANNELS = 3;   // 固定三通道
static map<string, double> g_timeAccum;  // 累積 ms

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

void calculate_distribution(int total_items, int size, vector<int>& counts, vector<int>& displs)
{
    counts.assign(size, total_items / size);
    int remainder = total_items % size;
    for (int i = 0; i < remainder; ++i) {
        counts[i]++;
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
}
void distributed_transpose(Mat& localMat, int global_rows, int global_cols)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1) Row distribution
    vector<int> row_counts(size), row_displs(size);
    calculate_distribution(global_rows, size, row_counts, row_displs);
    int local_rows = row_counts[rank];

    //  2) Column distribution(row distribution after transpose)
    vector<int> col_counts(size), col_displs(size);
    calculate_distribution(global_cols, size, col_counts, col_displs);
    int new_local_rows = col_counts[rank];

    // 3) Prepare send buffer
    vector<float> send_buf(local_rows * global_cols * 2);
    vector<int> send_counts(size), send_displs(size);
    int buf_idx = 0;

    for (int p = 0; p < size; ++p) {
        send_displs[p] = buf_idx;
        int p_cols_count = col_counts[p];
        int p_col_start = col_displs[p];
        
        for (int r = 0; r < local_rows; ++r) {
            const float* ptr = localMat.ptr<float>(r);
            for (int c = 0; c < p_cols_count; ++c) {
                int global_c = p_col_start + c;
                send_buf[buf_idx++] = ptr[global_c * 2];
                send_buf[buf_idx++] = ptr[global_c * 2 + 1];
            }
        }
        send_counts[p] = buf_idx - send_displs[p];
    }
    // 4) Prepare recv buffer
    vector<float> recv_buf(new_local_rows * global_rows * 2);
    vector<int> recv_counts(size), recv_displs(size);
    int recv_offset = 0;
    for (int p = 0; p < size; ++p) {
        recv_counts[p] = row_counts[p] * new_local_rows * 2;
        recv_displs[p] = recv_offset;
        recv_offset += recv_counts[p];
    }

    // 5) All-to-all communication
    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), MPI_FLOAT,
                  recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_FLOAT,
                  MPI_COMM_WORLD);
    
    // 6) Write back to localMat
    localMat = Mat::zeros(new_local_rows, global_rows, CV_32FC2);

    int current_idx = 0;
    int global_row_start = 0;
    for (int p = 0; p < size; ++p) {
        int src_rows = row_counts[p];
        for (int r = 0; r < src_rows; ++r) {
            for (int c = 0; c < new_local_rows; ++c) {
                float re = recv_buf[current_idx++];
                float im = recv_buf[current_idx++];
                localMat.at<Vec2f>(c, global_row_start + r) = Vec2f(re, im);
            }
        }
        global_row_start += src_rows;
    }

}

void my_dft2D(Mat& complexMat, int global_rows, int global_cols, bool inverse)
{
    CV_Assert(complexMat.type() == CV_32FC2);
    
    // 1) Row-wise transform (Local)
    for (int r = 0; r < complexMat.rows; ++r) {
        Vec2f* rowPtr = complexMat.ptr<Vec2f>(r);
        transform_row_inplace(rowPtr, global_cols, inverse);
    }

    // 2) Distributed transpose
    distributed_transpose(complexMat, global_rows, global_cols);
    
    // 3) Column-wise transform (Local)
    for (int r = 0; r < complexMat.rows; ++r) {
        Vec2f* rowPtr = complexMat.ptr<Vec2f>(r);
        transform_row_inplace(rowPtr, global_rows, inverse);
    }

    // 4) Distributed transpose back
    distributed_transpose(complexMat, global_cols, global_rows);
}

Mat wienerDeblur_myfft(const Mat& img, const Mat& psf, float K) {
    // call count
    if (callCount == 0) {
        g_timeAccum.clear();
    }
    callCount++;

    // MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int global_rows, global_cols, orig_rows, orig_cols;

    if (rank == 0) {
        orig_rows = img.rows;
        orig_cols = img.cols;
        global_rows = getOptimalDFTSize(orig_rows);
        global_cols = getOptimalDFTSize(orig_cols);
    }

    // 1. Preprocessing
    auto t_start = high_resolution_clock::now();

    MPI_Bcast(&global_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&orig_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&orig_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> row_counts(size), row_displs(size);
    calculate_distribution(global_rows, size, row_counts, row_displs);
    int local_rows = row_counts[rank];

    Mat local_complex_img(local_rows, global_cols, CV_32FC2);
    Mat local_complex_psf(local_rows, global_cols, CV_32FC2);

    vector<float> send_buf_img, send_buf_psf;
    vector<int> scats(size), displs(size);

    if (rank == 0) {
        Mat padded_img, padded_psf;
        copyMakeBorder(img, padded_img, 0, global_rows - img.rows, 0, global_cols - img.cols, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(psf, padded_psf, 0, global_rows - psf.rows, 0, global_cols - psf.cols, BORDER_CONSTANT, Scalar::all(0));
        
        Mat planes[] = {padded_img, Mat::zeros(padded_img.size(), CV_32F)};
        Mat complex_img_full;
        merge(planes, 2, complex_img_full);

        Mat psf_planes[] = {padded_psf, Mat::zeros(padded_psf.size(), CV_32F)};
        Mat complex_psf_full;
        merge(psf_planes, 2, complex_psf_full);
        
        // Prepare send buffers for scattering
        send_buf_img.assign((float*)complex_img_full.datastart, (float*)complex_img_full.dataend);
        send_buf_psf.assign((float*)complex_psf_full.datastart, (float*)complex_psf_full.dataend);

        for (int i = 0; i < size; ++i) {
            scats[i] = row_counts[i] * global_cols * 2;
            displs[i] = row_displs[i] * global_cols * 2;
        }
    }
    // Scatter image and PSF to all processes
    MPI_Scatterv(rank == 0 ? send_buf_img.data() : NULL, scats.data(), displs.data(), MPI_FLOAT,
                 local_complex_img.ptr<float>(), local_rows * global_cols * 2, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? send_buf_psf.data() : NULL, scats.data(), displs.data(), MPI_FLOAT,
                 local_complex_psf.ptr<float>(), local_rows * global_cols * 2, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    auto t_end = high_resolution_clock::now();
    double ms = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    g_timeAccum["MPI_simd: Pre-processing"] += ms;
    
    // 2. FFT Image
    t_start = high_resolution_clock::now();
    my_dft2D(local_complex_img, global_rows, global_cols, false);
    t_end = high_resolution_clock::now();
    ms = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    g_timeAccum["MPI_simd: FFT Image"] += ms;

    // 3. FFT PSF
    t_start = high_resolution_clock::now();
    my_dft2D(local_complex_psf, global_rows, global_cols, false);
    t_end = high_resolution_clock::now();
    ms = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    g_timeAccum["MPI_simd: FFT PSF"] += ms;

    // 4. Wiener Filter Calculation
    t_start = high_resolution_clock::now();
    // Compute |H|^2 + K (denominator) and H_conj
    // H (PSF), G (Image) -> F = G * (conj(H) / (|H|^2 + K))
    for (int r = 0; r < local_rows; ++r) {
        Vec2f* rowG = local_complex_img.ptr<Vec2f>(r);
        Vec2f* rowH = local_complex_psf.ptr<Vec2f>(r);
        for (int c = 0; c < global_cols; ++c) {
            complex<float> G(rowG[c][0], rowG[c][1]);
            complex<float> H(rowH[c][0], rowH[c][1]);

            float mag2 = norm(H); // |H|^2
            complex<float> H_conj = conj(H);
            complex<float> val = G * (H_conj / (mag2 + K));

            rowG[c][0] = val.real();
            rowG[c][1] = val.imag();
        }
    }
    t_end = high_resolution_clock::now();
    ms = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    g_timeAccum["MPI_simd: Wiener Filter"] += ms;

    // 5. Inverse FFT
    t_start = high_resolution_clock::now();
    my_dft2D(local_complex_img, global_rows, global_cols, true);
    t_end = high_resolution_clock::now();
    ms = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    g_timeAccum["MPI_simd: IFFT"] += ms;

    // 6. Post-processing
    t_start = high_resolution_clock::now();

    Mat finalRestored;
    if (rank == 0) {
        finalRestored = Mat::zeros(global_rows, global_cols, CV_32FC2);
    }

    MPI_Gatherv(local_complex_img.ptr<float>(), local_rows * global_cols * 2, MPI_FLOAT,
                rank == 0 ? (float*)finalRestored.data : NULL, scats.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        Mat full_restored = finalRestored;
        
        vector<Mat> planes;
        split(full_restored, planes);
        Mat restored = planes[0];
        restored /= (float)(global_rows * global_cols);

        // Crop
        finalRestored = restored(Rect(0, 0, orig_cols, orig_rows)).clone();
        normalize(finalRestored, finalRestored, 0, 1, NORM_MINMAX);
        
    }
    t_end = high_resolution_clock::now();
    ms = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
    g_timeAccum["MPI_simd: Post-processing"] += ms;

    if (rank == 0 && callCount == CHANNELS) {
        cout << "=== Accumulated Time ===" << endl;
        float this_round_total = 0;
        for (auto& p : g_timeAccum) {
            cout << p.first << " total: " << p.second << " ms" << endl;
            this_round_total += p.second;
        }
        cout << "this round total: " << this_round_total << " ms" << endl;
        cout << "=========================" << endl;
    }
    return finalRestored;
}

} // namespace fft_mpi_simd