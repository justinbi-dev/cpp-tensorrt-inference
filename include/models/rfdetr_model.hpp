#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "infer/inference_result.hpp"
#include "models/imodel.hpp"
#include "models/performance_metrics.hpp"
#include "utils/config.hpp"

class RFDETRModel : public IDetectorModel {
public:
    RFDETRModel(const DetectorConfig& cfg);
    std::vector<Detection> detect(const cv::Mat& frame) override;
    ~RFDETRModel() = default;

    const PerformanceMetrics& metrics() const { return metrics_; }
    const std::vector<std::string>& classNames() const { return classNames_; }

private:
    cv::Mat preprocess_(const cv::Mat& frame) override;
    const InferenceResult& infer_(const cv::Mat& frame) override;
    std::vector<Detection> postprocess_(const InferenceResult& inference_result, int sourceWidth,
                                        int sourceHeight) override;

    float confThresh_;
    int modelInputWidth_, modelInputHeight_, numClasses_, numQueries_;
    std::vector<std::string> classNames_;
    PerformanceMetrics metrics_;
};