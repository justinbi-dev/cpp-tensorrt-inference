#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "infer/inference_result.hpp"
#include "infer/trt_infer.hpp"
#include "models/detection.hpp"
#include "models/performance_metrics.hpp"
#include "utils/config.hpp"

class RFDETRModel {
public:
    explicit RFDETRModel(const DetectorConfig& cfg);
    std::vector<Detection> detect(const cv::Mat& frame);
    ~RFDETRModel() = default;

    const PerformanceMetrics& metrics() const { return metrics_; }
    const std::vector<std::string>& classNames() const { return classNames_; }

private:
    cv::Mat preprocess_(const cv::Mat& frame);
    const InferenceResult& infer_(const cv::Mat& frame);
    std::vector<Detection> postprocess_(const InferenceResult& inference_result, int sourceWidth,
                                        int sourceHeight);

    float confThresh_;
    int modelInputWidth_, modelInputHeight_, numClasses_, numQueries_;
    std::unique_ptr<TRTInfer> trt_infer_;
    std::vector<std::string> classNames_;
    PerformanceMetrics metrics_;
};