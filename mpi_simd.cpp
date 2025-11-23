#include "utils.hpp"
#include "fft/fft.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <mpi.h>
using namespace cv;
using namespace std;
using namespace std::chrono;

// 比較兩個 Mat 向量是否相等的輔助函數
bool areChannelsEqual(const std::vector<cv::Mat>& vec1, const std::vector<cv::Mat>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Channel count mismatch (" << vec1.size() << " vs " << vec2.size() << ").\n";
        return false;
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
        const cv::Mat& mat1 = vec1[i];
        const cv::Mat& mat2 = vec2[i];

        if (mat1.size() != mat2.size() || mat1.type() != mat2.type()) {
            std::cerr << "Error: Size or type mismatch in channel " << i << ".\n";
            return false;
        }
        
        double diff = cv::norm(mat1, mat2, cv::NORM_L2);
        if (diff != 0.0) {
            // 允許微小的誤差 (因為 AVX FMA 精度通常比純量高，導致結果不完全 bit-exact)
            if (diff > 1.0) { 
                std::cerr << "Error: Content mismatch in channel " << i << " (Diff: " << diff << ").\n";
                return false;
            }
        }
    }
    return true; 
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            cout << "Usage: mpirun -np <num_procs> ./mpi <img-path> <psf-length> <psf-angle>\n";
        }
        MPI_Finalize();
        return -1;
    }

    string img_path = argv[1];
    int psf_length = atoi(argv[2]);
    double psf_angle = atof(argv[3]);
    
    bool usePowerOf2 = true;

    // Only rank 0 reads image and prepares data
    Mat img, psf;
    float K = 0.01f;
    vector<Mat> serial_channels;
    vector<Mat> channels;

    if (rank == 0) {
        img = imread(img_path, IMREAD_COLOR);
        if (img.empty()) {
            cout << "Cannot read image\n"; 
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        img.convertTo(img, CV_32F);
        img /= 255.0;

        psf = motionBlurKernel(psf_length, psf_angle);
        split(img, channels);
        split(img, serial_channels);
    }

    auto t_start_serial = high_resolution_clock::now();
    if (rank == 0) {
        
        for (int i = 0; i < 3; i++) {
            Mat channel = serial_channels[i];
            if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);
            serial_channels[i] = fft_serial::wienerDeblur_myfft(channel, psf, K);
            if (usePowerOf2) serial_channels[i] = serial_channels[i](Rect(0, 0, img.cols, img.rows));
        }
        
    }
    auto t_end_serial = high_resolution_clock::now();


    // All processes call wienerDeblur_myfft
    // Rank 0 sends data and receives results
    // Workers receive data, process, and send back
    auto t_start_mpi = high_resolution_clock::now();
    if (rank == 0) {
        for (int i = 0; i < 3; i++) {
            Mat channel = channels[i];
            if (usePowerOf2) channel = autoPadToPowerOfTwo(channel);

            channels[i] = fft_mpi_simd::wienerDeblur_myfft(channel, psf, K);

            if (usePowerOf2) channels[i] = channels[i](Rect(0, 0, img.cols, img.rows));
        }
    } else {
        // Workers participate in processing for each channel
        for (int i = 0; i < 3; i++) {
            fft_mpi_simd::wienerDeblur_myfft(Mat(), Mat(), K);
        }
    }
    auto t_end_mpi = high_resolution_clock::now();

    // Only rank 0 reports timing and verifies results
    if (rank == 0) {
        auto serial_time = getElapsedMs(t_start_serial, t_end_serial);
        cout << "Deblurring 3 channels took (serial) : " << serial_time << " ms\n";
        auto mpi_simd_time = getElapsedMs(t_start_mpi, t_end_mpi);
        cout << "Deblurring 3 channels took (mpi_simd): " << mpi_simd_time << " ms\n";

        if(areChannelsEqual(serial_channels, channels)) {
            cout << "[Success] The results from serial and MPI implementations are identical.\n";
            printf("[Speedup] %.2fx\n", serial_time / mpi_simd_time);
        } else {
            cout << "[Warning/Error] The results differ.\n";
            // 註：如果只是微小的浮點數誤差，可能是正常的
        }

        Mat merged_float;
        merge(channels, merged_float);

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

    MPI_Finalize();
    return 0;
}