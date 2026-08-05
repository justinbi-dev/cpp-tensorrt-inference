#pragma once
#include "idetector.hpp"
#include <NvInfer.h>
#include <memory>

class TRTDetector : public IDetector {
public:
    TRTDetector(const std::string& enginePath,
                float confThresh = 0.25f,
                float nmsThresh = 0.45f,
                cv::Size inputSize = {640, 640});
    ~TRTDetector() override;

    std::vector<Detection> detect(const cv::Mat& frame) override;
    cv::Size inputSize() const override { return inputSize_; }

private:
    void loadEngine(const std::string& path);
    void allocateBuffers();

    cv::Size inputSize_;
    float confThresh_, nmsThresh_;

    // TensorRT objects — use unique_ptr with custom deleter or nvinfer1's own
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    void* inputDevice_ = nullptr;
    void* outputDevice_ = nullptr;
    std::vector<float> outputHost_;
    cudaStream_t stream_;

    std::string inputName_, outputName_;
    size_t inputSize_bytes_ = 0;
    size_t outputSize_bytes_ = 0;
    nvinfer1::Dims outputDims_;
};