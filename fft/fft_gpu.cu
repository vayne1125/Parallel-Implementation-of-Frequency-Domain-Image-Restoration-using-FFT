#include "fft.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "../utils.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono; // 為了 CpuTimer

namespace fft_gpu {

#define PI 3.14159265358979323846f

struct Profiler {
    float t_alloc = 0;
    float t_h2d = 0;
    float t_pre = 0;
    float t_compute = 0; // FFT + Filter + IFFT
    float t_d2h = 0;
    float t_post = 0;
    
    cudaEvent_t start, stop;
    cudaStream_t stream;

    Profiler(cudaStream_t s) : stream(s) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~Profiler() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tick() { cudaEventRecord(start, stream); }
    void tock(float& target) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        target += ms;
    }
    
    void print(string title) {
        cout << "=== " << title << " Profiling (3 Channels) ===" << endl;
        cout << "[1. Allocation]  Time: " << t_alloc   << " ms (MallocHost + Malloc)" << endl;
        cout << "[2. H2D Copy]    Time: " << t_h2d     << " ms (Raw Img + PSF + Twiddle)" << endl;
        cout << "[3. Pre-process] Time: " << t_pre     << " ms (Padding + PSF FFT)" << endl;
        cout << "[4. GPU Compute] Time: " << t_compute << " ms (FFT + Filter + IFFT)" << endl;
        cout << "[5. D2H Copy]    Time: " << t_d2h     << " ms (Result Transfer)" << endl;
        cout << "[6. Post-process]Time: " << t_post    << " ms (Normalize + CPU Copy)" << endl;
        cout << "--------------------------------------------" << endl;
        cout << "Total (Sum)      Time: " << (t_alloc + t_h2d + t_pre + t_compute + t_d2h + t_post) << " ms" << endl;
        cout << "============================================" << endl;
    }
};

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// ============================================================
// Complex Math Helpers
// ============================================================
__device__ inline float2 complex_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ inline float2 complex_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ inline float2 complex_sub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// ============================================================
// Kernel: Preprocess (BGR uchar -> Complex float with Padding)
// 同時完成：型別轉換、Padding、設虛部為0
// ============================================================
__global__ void preprocess_kernel(const float* src_img, float2* dst_complex, 
                                  int src_rows, int src_cols, 
                                  int dst_rows, int dst_cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_cols && y < dst_rows) {
        int idx = y * dst_cols + x;
        
        if (x < src_cols && y < src_rows) {
            // 有效區域：讀取像素值 (假設已經正規化為 0-1 的 float)
            float pixel = src_img[y * src_cols + x];
            dst_complex[idx] = make_float2(pixel, 0.0f);
        } else {
            // Padding 區域：補 0
            dst_complex[idx] = make_float2(0.0f, 0.0f);
        }
    }
}

// ============================================================
// Kernel: FFT Row
// ============================================================
__global__ void fft_row_optimized_kernel(float2* d_data, const float2* d_twiddles, int M, int N, int logN, bool inverse) {
    int row = blockIdx.x;
    if (row >= M) return;
    extern __shared__ float2 s_data[];
    int tid = threadIdx.x;
    
    // Load
    for (int i = tid; i < N; i += blockDim.x) s_data[i] = d_data[row * N + i];
    __syncthreads();

    // Bit Reversal
    for (int i = tid; i < N; i += blockDim.x) {
        int rev = 0, temp = i;
        for (int k = 0; k < logN; ++k) { rev = (rev << 1) | (temp & 1); temp >>= 1; }
        if (i < rev) { float2 t = s_data[i]; s_data[i] = s_data[rev]; s_data[rev] = t; }
    }
    __syncthreads();

    // Stages
    for (int len = 2; len <= N; len <<= 1) {
        int half_len = len >> 1;
        int twiddle_step = N / len; 
        for (int k = tid; k < N / 2; k += blockDim.x) {
             int offset = k & (half_len - 1);
             int u_idx = ((k - offset) << 1) + offset;
             int v_idx = u_idx + half_len;
             int t_idx = offset * twiddle_step;
             float2 w = d_twiddles[t_idx];
             if (inverse) w.y = -w.y; 
             float2 u = s_data[u_idx];
             float2 v = s_data[v_idx];
             float2 vw = complex_mul(v, w);
             s_data[u_idx] = complex_add(u, vw);
             s_data[v_idx] = complex_sub(u, vw);
        }
        __syncthreads();
    }
    
    // Store
    for (int i = tid; i < N; i += blockDim.x) d_data[row * N + i] = s_data[i];
}

// ============================================================
// Kernel: Transpose
// ============================================================
const int TILE_DIM = 32;
__global__ void transpose_kernel_opt(float2* src, float2* dst, int rows, int cols) {
    __shared__ float2 tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = src[y * cols + x];
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    // 轉置寫回
    if (x < rows && y < cols) dst[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

// ============================================================
// Kernel: Wiener Filter
// ============================================================
__global__ void wiener_kernel(float2* d_img, const float2* d_psf, int total_pixels, float K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float2 G = d_img[idx];
        float2 H = d_psf[idx];
        float mag2 = H.x * H.x + H.y * H.y;
        float denom = mag2 + K;
        float invDenom = (denom > 1e-8f) ? (1.0f / denom) : 0.0f;
        float2 H_conj = make_float2(H.x, -H.y);
        float2 num = complex_mul(G, H_conj);
        d_img[idx] = make_float2(num.x * invDenom, num.y * invDenom);
    }
}

// ============================================================
// Kernel: Post-process (Extract Real & Scale)
// 直接輸出 Crop 後的有效區域，減少傳輸量
// ============================================================
__global__ void postprocess_kernel(float2* src_complex, float* dst_real, 
                                   int src_rows, int src_cols, 
                                   int dst_rows, int dst_cols, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 只處理有效區域 (Crop)
    if (x < dst_cols && y < dst_rows) {
        int src_idx = y * src_cols + x;
        int dst_idx = y * dst_cols + x;
        
        // 取實部並縮放
        dst_real[dst_idx] = src_complex[src_idx].x * scale;
    }
}

// ============================================================
// Helpers & Driver
// ============================================================
void generate_twiddles(float2* twiddles, int N) {
    for (int k = 0; k < N / 2; ++k) {
        double angle = -2.0 * CV_PI * k / N;
        twiddles[k].x = cos(angle);
        twiddles[k].y = sin(angle);
    }
}

void my_dft2D_gpu_impl(float2* d_data, float2* d_temp, float2* d_twiddles_rows, float2* d_twiddles_cols, int M, int N, bool inverse) {
    int logN = int(log2((double)N));
    int logM = int(log2((double)M));
    
    // Row FFT
    size_t shared_mem = N * sizeof(float2);
    int threads = (N/2 > 1024) ? 1024 : (N/2); if (threads < 1) threads = 1;
    fft_row_optimized_kernel<<<M, threads, shared_mem>>>(d_data, d_twiddles_rows, M, N, logN, inverse);
    CHECK_CUDA(cudaGetLastError());

    // Transpose
    dim3 dimBlock(32, 32);
    dim3 dimGrid1((N + 31) / 32, (M + 31) / 32);
    transpose_kernel_opt<<<dimGrid1, dimBlock>>>(d_data, d_temp, M, N);
    CHECK_CUDA(cudaGetLastError());

    // Col FFT
    size_t shared_mem_col = M * sizeof(float2);
    int threads_col = (M/2 > 1024) ? 1024 : (M/2); if (threads_col < 1) threads_col = 1;
    fft_row_optimized_kernel<<<N, threads_col, shared_mem_col>>>(d_temp, d_twiddles_cols, N, M, logM, inverse);
    CHECK_CUDA(cudaGetLastError());

    // Transpose Back
    dim3 dimGrid2((M + 31) / 32, (N + 31) / 32);
    transpose_kernel_opt<<<dimGrid2, dimBlock>>>(d_temp, d_data, N, M);
    CHECK_CUDA(cudaGetLastError());
}

Mat precomputePSF(const Mat& psf, int optRows, int optCols) {
    // PSF 只需要做一次，用 OpenCV 準備資料沒關係
    Mat psfPadded;
    copyMakeBorder(psf, psfPadded, 0, optRows - psf.rows, 0, optCols - psf.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {psfPadded, Mat::zeros(psfPadded.size(), CV_32F)};
    Mat psfComplex;
    merge(planes, 2, psfComplex);

    vector<float2> tw_rows(optCols / 2);
    vector<float2> tw_cols(optRows / 2);
    generate_twiddles(tw_rows.data(), optCols);
    generate_twiddles(tw_cols.data(), optRows);

    size_t bytes = optRows * optCols * sizeof(float2);
    float2 *d_psf, *d_temp, *d_tw_r, *d_tw_c;
    
    CHECK_CUDA(cudaMalloc(&d_psf, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp, bytes));
    CHECK_CUDA(cudaMalloc(&d_tw_r, tw_rows.size() * sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&d_tw_c, tw_cols.size() * sizeof(float2)));

    CHECK_CUDA(cudaMemcpy(d_psf, psfComplex.data, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tw_r, tw_rows.data(), tw_rows.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tw_c, tw_cols.data(), tw_cols.size() * sizeof(float2), cudaMemcpyHostToDevice));

    my_dft2D_gpu_impl(d_psf, d_temp, d_tw_r, d_tw_c, optRows, optCols, false);

    CHECK_CUDA(cudaMemcpy(psfComplex.data, d_psf, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_psf); cudaFree(d_temp); cudaFree(d_tw_r); cudaFree(d_tw_c);
    return psfComplex;
}

// ============================================================
// Version A: Fast Version (Allocation Reuse)
// 配置移到迴圈外，只做一次
// ============================================================
void wienerDeblur_RGB_optimized(vector<Mat>& channels, const Mat& psf, float K) {
    // 建立 Stream 用於計時
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    Profiler p(stream);

    int imgRows = channels[0].rows;
    int imgCols = channels[0].cols;
    int optRows = nextPowerOfTwo(imgRows); 
    int optCols = nextPowerOfTwo(imgCols);
    int psfRows = psf.rows;
    int psfCols = psf.cols;

    size_t bytes_raw_img = imgRows * imgCols * sizeof(float);
    size_t bytes_raw_psf = psfRows * psfCols * sizeof(float);
    size_t bytes_complex = optRows * optCols * sizeof(float2);

    // 1. Allocation (Outside Loop)
    float *h_pinned_img_in, *h_pinned_img_out, *h_pinned_psf_in;
    float *d_img_in_raw, *d_img_out_raw, *d_psf_in_raw;
    float2 *d_complex_img, *d_complex_psf, *d_temp;
    float2 *d_tw_r, *d_tw_c;
    vector<float2> tw_rows(optCols / 2);
    vector<float2> tw_cols(optRows / 2);

    p.tick(); // Start Alloc
    {
        CHECK_CUDA(cudaMallocHost((void**)&h_pinned_img_in, bytes_raw_img));
        CHECK_CUDA(cudaMallocHost((void**)&h_pinned_img_out, bytes_raw_img));
        CHECK_CUDA(cudaMallocHost((void**)&h_pinned_psf_in, bytes_raw_psf));

        CHECK_CUDA(cudaMalloc(&d_img_in_raw, bytes_raw_img));
        CHECK_CUDA(cudaMalloc(&d_img_out_raw, bytes_raw_img));
        CHECK_CUDA(cudaMalloc(&d_psf_in_raw, bytes_raw_psf));
        CHECK_CUDA(cudaMalloc(&d_complex_img, bytes_complex));
        CHECK_CUDA(cudaMalloc(&d_complex_psf, bytes_complex));
        CHECK_CUDA(cudaMalloc(&d_temp, bytes_complex));

        generate_twiddles(tw_rows.data(), optCols);
        generate_twiddles(tw_cols.data(), optRows);
        CHECK_CUDA(cudaMalloc(&d_tw_r, tw_rows.size() * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&d_tw_c, tw_cols.size() * sizeof(float2)));
    }
    p.tock(p.t_alloc); // End Alloc

    // 迴圈處理 RGB
    for (int i = 0; i < channels.size(); ++i) {
        Mat& img = channels[i];

        // Part A: PSF & Twiddle (H2D + Preprocess)
        // 模擬 Serial 行為：每次都重做 PSF，算入時間
        p.tick();
        memcpy(h_pinned_psf_in, psf.data, bytes_raw_psf);
        CHECK_CUDA(cudaMemcpyAsync(d_tw_r, tw_rows.data(), tw_rows.size() * sizeof(float2), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_tw_c, tw_cols.data(), tw_cols.size() * sizeof(float2), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_psf_in_raw, h_pinned_psf_in, bytes_raw_psf, cudaMemcpyHostToDevice, stream));
        p.tock(p.t_h2d);

        p.tick();
        dim3 block(32, 32);
        dim3 grid_psf((optCols + 31) / 32, (optRows + 31) / 32);
        preprocess_kernel<<<grid_psf, block, 0, stream>>>(d_psf_in_raw, d_complex_psf, psfRows, psfCols, optRows, optCols);
        dim3 grid_img((optCols + 31) / 32, (optRows + 31) / 32);
        preprocess_kernel<<<grid_img, block, 0, stream>>>(d_img_in_raw, d_complex_img, imgRows, imgCols, optRows, optCols);
        p.tock(p.t_pre); // PSF FFT 算在 Pre-process 裡
        
        // Part B: Image (H2D)
        p.tick();
        if (img.isContinuous()) memcpy(h_pinned_img_in, img.data, bytes_raw_img);
        else { Mat cont = img.clone(); memcpy(h_pinned_img_in, cont.data, bytes_raw_img); }
        CHECK_CUDA(cudaMemcpyAsync(d_img_in_raw, h_pinned_img_in, bytes_raw_img, cudaMemcpyHostToDevice, stream));
        p.tock(p.t_h2d);
        
        // Part C: Image Compute (Pre + FFT + Filter + IFFT + Post)
        // 我們把 Preprocess Image 也算進 Compute 以簡化
        p.tick(); 
        
        my_dft2D_gpu_impl(d_complex_psf, d_temp, d_tw_r, d_tw_c, optRows, optCols, false);
        my_dft2D_gpu_impl(d_complex_img, d_temp, d_tw_r, d_tw_c, optRows, optCols, false); // FFT

        int total_pixels = optRows * optCols;
        int threads = 256;
        int blocks = (total_pixels + threads - 1) / threads;
        wiener_kernel<<<blocks, threads, 0, stream>>>(d_complex_img, d_complex_psf, total_pixels, K); // Filter

        my_dft2D_gpu_impl(d_complex_img, d_temp, d_tw_r, d_tw_c, optRows, optCols, true); // IFFT

        float scale = 1.0f / (float)(optRows * optCols);
        dim3 grid_post((imgCols + 31) / 32, (imgRows + 31) / 32);
        postprocess_kernel<<<grid_post, block, 0, stream>>>(d_complex_img, d_img_out_raw, optRows, optCols, imgRows, imgCols, scale);
        p.tock(p.t_compute);

        // Part D: D2H
        p.tick();
        CHECK_CUDA(cudaMemcpyAsync(h_pinned_img_out, d_img_out_raw, bytes_raw_img, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        p.tock(p.t_d2h);

        // Part E: CPU Post
        p.tick();
        Mat finalRestored(imgRows, imgCols, CV_32F);
        memcpy(finalRestored.data, h_pinned_img_out, bytes_raw_img);
        normalize(finalRestored, finalRestored, 0, 1, NORM_MINMAX);
        p.tock(p.t_post);
        
        channels[i] = finalRestored;
    }

    p.print("FAST (Reuse Memory)");

    cudaStreamDestroy(stream);
    cudaFree(d_img_in_raw); cudaFree(d_img_out_raw); cudaFree(d_psf_in_raw);
    cudaFree(d_complex_img); cudaFree(d_complex_psf); cudaFree(d_temp);
    cudaFree(d_tw_r); cudaFree(d_tw_c);
    cudaFreeHost(h_pinned_img_in); cudaFreeHost(h_pinned_img_out); cudaFreeHost(h_pinned_psf_in);
};

// ============================================================
// Version B: Slow Version (Simulate Allocation Inside Loop)
// 這就是你原本覺得慢的版本，用來對照
// ============================================================
void wienerDeblur_RGB_naive(vector<Mat>& channels, const Mat& psf, float K) {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    Profiler p(stream); // 使用同一個 Profiler class

    // ... 尺寸計算同上 ...
    int imgRows = channels[0].rows;
    int imgCols = channels[0].cols;
    int optRows = nextPowerOfTwo(imgRows); 
    int optCols = nextPowerOfTwo(imgCols);
    int psfRows = psf.rows;
    int psfCols = psf.cols;

    size_t bytes_raw_img = imgRows * imgCols * sizeof(float);
    size_t bytes_raw_psf = psfRows * psfCols * sizeof(float);
    size_t bytes_complex = optRows * optCols * sizeof(float2);

    // 迴圈處理 RGB (注意：所有 malloc 都在裡面)
    for (int i = 0; i < channels.size(); ++i) {
        Mat& img = channels[i];
        
        float *h_pinned_img_in, *h_pinned_img_out, *h_pinned_psf_in;
        float *d_img_in_raw, *d_img_out_raw, *d_psf_in_raw;
        float2 *d_complex_img, *d_complex_psf, *d_temp;
        float2 *d_tw_r, *d_tw_c;
        vector<float2> tw_rows(optCols / 2);
        vector<float2> tw_cols(optRows / 2);

        // [差異點] Allocation Inside Loop
        p.tick();
        {
            CHECK_CUDA(cudaMallocHost((void**)&h_pinned_img_in, bytes_raw_img));
            CHECK_CUDA(cudaMallocHost((void**)&h_pinned_img_out, bytes_raw_img));
            CHECK_CUDA(cudaMallocHost((void**)&h_pinned_psf_in, bytes_raw_psf));

            CHECK_CUDA(cudaMalloc(&d_img_in_raw, bytes_raw_img));
            CHECK_CUDA(cudaMalloc(&d_img_out_raw, bytes_raw_img));
            CHECK_CUDA(cudaMalloc(&d_psf_in_raw, bytes_raw_psf));
            CHECK_CUDA(cudaMalloc(&d_complex_img, bytes_complex));
            CHECK_CUDA(cudaMalloc(&d_complex_psf, bytes_complex));
            CHECK_CUDA(cudaMalloc(&d_temp, bytes_complex));

            generate_twiddles(tw_rows.data(), optCols);
            generate_twiddles(tw_cols.data(), optRows);
            CHECK_CUDA(cudaMalloc(&d_tw_r, tw_rows.size() * sizeof(float2)));
            CHECK_CUDA(cudaMalloc(&d_tw_c, tw_cols.size() * sizeof(float2)));
        }
        p.tock(p.t_alloc);

        // ... (以下運算邏輯與 Fast Version 完全相同，只是變數是區域的) ...
        // [H2D]
        p.tick();
        memcpy(h_pinned_psf_in, psf.data, bytes_raw_psf);
        CHECK_CUDA(cudaMemcpyAsync(d_tw_r, tw_rows.data(), tw_rows.size() * sizeof(float2), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_tw_c, tw_cols.data(), tw_cols.size() * sizeof(float2), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_psf_in_raw, h_pinned_psf_in, bytes_raw_psf, cudaMemcpyHostToDevice, stream));
        
        if (img.isContinuous()) memcpy(h_pinned_img_in, img.data, bytes_raw_img);
        else { Mat cont = img.clone(); memcpy(h_pinned_img_in, cont.data, bytes_raw_img); }
        CHECK_CUDA(cudaMemcpyAsync(d_img_in_raw, h_pinned_img_in, bytes_raw_img, cudaMemcpyHostToDevice, stream));
        p.tock(p.t_h2d);

        // [Pre-process]
        p.tick();
        dim3 block(32, 32);
        dim3 grid_psf((optCols + 31) / 32, (optRows + 31) / 32);
        dim3 grid_img((optCols + 31) / 32, (optRows + 31) / 32);
        preprocess_kernel<<<grid_psf, block, 0, stream>>>(d_psf_in_raw, d_complex_psf, psfRows, psfCols, optRows, optCols);
        preprocess_kernel<<<grid_img, block, 0, stream>>>(d_img_in_raw, d_complex_img, imgRows, imgCols, optRows, optCols);
        p.tock(p.t_pre);
        
        // [Compute]
        p.tick();
        my_dft2D_gpu_impl(d_complex_psf, d_temp, d_tw_r, d_tw_c, optRows, optCols, false);
        my_dft2D_gpu_impl(d_complex_img, d_temp, d_tw_r, d_tw_c, optRows, optCols, false);
    
        int total_pixels = optRows * optCols;
        int threads = 256;
        int blocks = (total_pixels + threads - 1) / threads;
        wiener_kernel<<<blocks, threads, 0, stream>>>(d_complex_img, d_complex_psf, total_pixels, K);

        my_dft2D_gpu_impl(d_complex_img, d_temp, d_tw_r, d_tw_c, optRows, optCols, true);

        float scale = 1.0f / (float)(optRows * optCols);
        dim3 grid_post((imgCols + 31) / 32, (imgRows + 31) / 32);
        postprocess_kernel<<<grid_post, block, 0, stream>>>(d_complex_img, d_img_out_raw, optRows, optCols, imgRows, imgCols, scale);
        p.tock(p.t_compute);

        // [D2H]
        p.tick();
        CHECK_CUDA(cudaMemcpyAsync(h_pinned_img_out, d_img_out_raw, bytes_raw_img, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        p.tock(p.t_d2h);

        // [Post]
        p.tick();
        Mat finalRestored(imgRows, imgCols, CV_32F);
        memcpy(finalRestored.data, h_pinned_img_out, bytes_raw_img);
        normalize(finalRestored, finalRestored, 0, 1, NORM_MINMAX);
        p.tock(p.t_post);
        
        channels[i] = finalRestored;

        // [Free Inside Loop]
        cudaFree(d_img_in_raw); cudaFree(d_img_out_raw); cudaFree(d_psf_in_raw);
        cudaFree(d_complex_img); cudaFree(d_complex_psf); cudaFree(d_temp);
        cudaFree(d_tw_r); cudaFree(d_tw_c);
        cudaFreeHost(h_pinned_img_in); cudaFreeHost(h_pinned_img_out); cudaFreeHost(h_pinned_psf_in);
    }

    p.print("SLOW (Naive Allocation)");
    cudaStreamDestroy(stream);
}

void fft_radix2_kernel(float* data, int n, bool inverse) {} 
void my_dft2D(Mat& complexMat, bool inverse) {}

} // namespace fft_gpu