#pragma once
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

#include "infer/inference_result.hpp"

struct TensorInfo {
    std::string name;
    bool is_input;
    std::vector<int64_t> shape;
    nvinfer1::DataType dtype;
    size_t bytes;
};

class TRTInfer {
public:
    TRTInfer(const std::string& path);
    ~TRTInfer();
    TRTInfer(const TRTInfer&) = delete;
    TRTInfer& operator=(const TRTInfer&) = delete;

    // InferenceResult is only valid until the next infer() call
    const InferenceResult& infer(const cv::Mat& blob);
    const TensorInfo& inputTensorInfo() const;
    const TensorInfo& outputTensorInfo(const std::string& name) const;

private:
    void loadEngine(const std::string& path);
    void allocateBuffers();

    struct TensorBinding{
        TensorInfo info;
        void* device_buffer = nullptr;
    };

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<TensorBinding> tensors_;
    std::vector<size_t> input_indices_;
    std::vector<size_t> output_indices_;

    InferenceResult output_;
    cudaStream_t stream_;

};