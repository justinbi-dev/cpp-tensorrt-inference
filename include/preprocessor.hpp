#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat preprocess(const cv::Mat& frame, cv::Size targetSize);