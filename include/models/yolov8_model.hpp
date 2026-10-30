#pragma once
#include "infer/inference_result.hpp"
#include "models/performance_metrics.hpp"
#include "models/imodel.hpp"
#include "utils/config.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class YOLOv8Model: public IDetectorModel {
public:
    YOLOv8Model(const DetectorConfig& cfg);
    ~YOLOv8Model() = default;
      

private:
    cv::Mat preprocess_(const cv::Mat& frame) override;
    InferenceResult infer_(const cv::Mat& frame) override;
    std::vector<Detection> postprocess_(InferenceResult& inference_result) override;

    std::string enginePath;
    float confThresh_, nmsThresh_;
    uint modelInputWidth_, modelInputHeight_, sourceWidth_, sourceHeight_, numClasses_;
};