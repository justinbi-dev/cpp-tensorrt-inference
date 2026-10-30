#pragma once
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

#include "infer/inference_result.hpp"


class TRTInfer {
public:
    TRTInfer(const std::string& path);
    ~TRTInfer();

    InferenceResult infer(const cv::Mat& blob);

private:
    void loadEngine(const std::string& path);
    void allocateBuffers();

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::vector<void*> in_device_buffers_;
    std::vector<void*> out_device_buffers_;
    //std::vector<std::vector<float>> out_host_buffers_;
    std::vector<std::vector<int64_t>> out_shapes;
    std::vector<std::vector<int64_t>> in_shapes;

    InferenceResult output_;
    

    //void* inputDevice_ = nullptr;
    //void* outputDevice_ = nullptr;
    //std::vector<float> outputHost_;
    cudaStream_t stream_;

    std::string inputName_, outputName_;
    size_t inputSize_bytes_ = 0;
    size_t outputSize_bytes_ = 0;
    nvinfer1::Dims outputDims_;

};