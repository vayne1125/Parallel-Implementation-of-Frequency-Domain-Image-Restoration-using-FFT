#include "fft.hpp"
#include "../utils.hpp"
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
#include <mpi.h>
using namespace cv;
using namespace std;

namespace fft_mpi {
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

// 2D separable transform: row-wise FFT, transpose, row-wise FFT, transpose back
// works in-place on CV_32FC2 Mat
// inverse: false = forward DFT, true = inverse DFT (no scaling)
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
    // MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int global_rows, global_cols, orig_rows, orig_cols;

    // 1) Rank 0
    if (rank == 0) {
        orig_rows = img.rows;
        orig_cols = img.cols;
        global_rows = getOptimalDFTSize(orig_rows);
        global_cols = getOptimalDFTSize(orig_cols);
    }
    MPI_Bcast(&global_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&orig_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&orig_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> row_counts(size), row_displs(size);
    calculate_distribution(global_rows, size, row_counts, row_displs);
    int local_rows = row_counts[rank];

    // 2) Prepare local image Mat
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

    // 3) Forward DFT on local data
    my_dft2D(local_complex_img, global_rows, global_cols, false);
    my_dft2D(local_complex_psf, global_rows, global_cols, false);

    // 4) Compute |H|^2 + K (denominator) and H_conj
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

    // 5) Inverse DFT on local data
    my_dft2D(local_complex_img, global_rows, global_cols, true);

    // 6) Gather results to rank 0
    Mat finalRestored;
    if (rank == 0) {
        finalRestored = Mat::zeros(global_rows, global_cols, CV_32FC2);
    }

    MPI_Gatherv(local_complex_img.ptr<float>(), local_rows * global_cols * 2, MPI_FLOAT,
                rank == 0 ? (float*)finalRestored.data : NULL, scats.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    // 7) Rank 0: extract real part and crop to original size
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
    return finalRestored;
}
}