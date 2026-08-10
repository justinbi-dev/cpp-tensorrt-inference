#pragma once
#include "types.hpp"
#include "imodel.hpp"
#include "yolov8_model.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class YOLOv8Model: public IDetectorModel {
public:
    YOLOv8Model(const std::string& enginePath,
                float confThresh = 0.25f,
                float nmsThresh = 0.45f,
                uint modelInputWidth = 640,
                uint modelInputHeight = 640,
                uint sourceWidth = 1920,
                uint sourceHeight = 1080,
                uint numClasses = 80);

    virtual cv::Mat preprocess(const cv::Mat& frame);       
    virtual std::vector<Detection> postprocess(const std::vector<float>& rawOutput);

private:
    std::string enginePath;
    float confThresh_, nmsThresh_;
    uint modelInputWidth_, modelInputHeight_, sourceWidth_, sourceHeight_, numClasses_;
};