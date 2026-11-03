#pragma once
#include "models/detection.hpp"
#include "infer/trt_infer.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class IModel {
public:
    virtual ~IModel() = default;
protected:
    std::unique_ptr<TRTInfer> trt_infer_;
};

class IDetectorModel : public IModel {
public:
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;
    virtual ~IDetectorModel() = default;

private:      
    virtual cv::Mat preprocess_(const cv::Mat& frame) = 0;
    virtual InferenceResult infer_(const cv::Mat& frame) = 0;
    virtual std::vector<Detection> postprocess_(const InferenceResult& inference_result, int sourceWidth, int sourceHeight) = 0;
};