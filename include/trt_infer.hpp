#pragma once
#include "imodel.hpp"
#include "config.hpp"
#include "yolov8_model.hpp"
#include <NvInfer.h>
#include <memory>

class TRTInfer {
public:
    TRTInfer(const DetectorConfig cfg);
    ~TRTInfer();

    std::vector<Detection> detect(const cv::Mat& frame);

private:
    void loadEngine(const std::string& path);
    void allocateBuffers();

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

    std::unique_ptr<IDetectorModel> model_;
};