#pragma once
#include "types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class IModel {
public:
    virtual ~IModel() = default;
    virtual std::vector<float> preprocess(const cv::Mat& frame) = 0;
};

class IDetectorModel : public IModel {
public:
    virtual std::vector<Detection> postprocess(const std::vector<float>& rawOutput,
                                                 const cv::Size& originalFrameSize) = 0;
};