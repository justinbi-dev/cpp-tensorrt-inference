#pragma once
#include "infer/inference_result.hpp"
#include "models/performance_metrics.hpp"
#include "models/imodel.hpp"
#include "utils/config.hpp"
#include "utils/timer.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

class RFDETRModel: public IDetectorModel {
public:
    RFDETRModel(const DetectorConfig& cfg);
    std::vector<Detection> detect(const cv::Mat& frame) override;
    ~RFDETRModel() = default;

    const PerformanceMetrics& metrics() const {return metrics_; }
      

private:
    cv::Mat preprocess_(const cv::Mat& frame) override;
    const InferenceResult& infer_(const cv::Mat& frame) override;
    std::vector<Detection> postprocess_(const InferenceResult& inference_result, int sourceWidth, int sourceHeight) override;

    float confThresh_;
    int modelInputWidth_, modelInputHeight_, numClasses_, numQueries_;
    PerformanceMetrics metrics_;
};