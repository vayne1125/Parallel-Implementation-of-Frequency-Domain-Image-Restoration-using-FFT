#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace cv;
using namespace std;
using namespace std::chrono;

// 計算時間
double getElapsedMs(high_resolution_clock::time_point start,
                    high_resolution_clock::time_point end);

// Motion blur kernel
Mat motionBlurKernel(int size, double angle);

// 計算最接近 >= n 的 2 的冪次
int nextPowerOfTwo(int n);

// 自動 padding
Mat autoPadToPowerOfTwo(const Mat& src);

// 判斷是否為 2 的冪
bool isPowerOfTwo(int n);

// 白平衡校正
Mat applyWhiteBalance(const Mat& img_Lab, const Mat& img_orig_Lab);
