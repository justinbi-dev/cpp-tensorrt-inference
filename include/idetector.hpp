#pragma once
#include "types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;
    virtual cv::Size inputSize() const = 0;
};