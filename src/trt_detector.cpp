#include "trt_detector.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include <fstream>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static Logger gLogger;

TRTDetector::TRTDetector(const std::string& enginePath, float conf, float nms, cv::Size size)
    : inputSize_(size), confThresh_(conf), nmsThresh_(nms) {
    loadEngine(enginePath);
    allocateBuffers();
    cudaStreamCreate(&stream_);
}

void TRTDetector::loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    if (size == 0) {
        throw std::runtime_error("Engine file is empty: " + path);
    }
    file.seekg(0, std::ios::beg);

    std::vector<char> buf(size);
    if (!file.read(buf.data(), size)) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }

    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    engine_.reset(runtime_->deserializeCudaEngine(buf.data(), size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize engine — check TensorRT version match");
    }
    context_.reset(engine_->createExecutionContext());
}
void TRTDetector::allocateBuffers() {
    int numIO = engine_->getNbIOTensors();

    for (int i = 0; i < numIO; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        nvinfer1::Dims dims = engine_->getTensorShape(name);

        size_t elemCount = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            elemCount *= dims.d[d];
        }
        size_t bytes = elemCount * sizeof(float);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputName_ = name;
            inputSize_bytes_ = bytes;
            cudaMalloc(&inputDevice_, bytes);
        } else {
            outputName_ = name;
            outputSize_bytes_ = bytes;
            outputDims_ = dims;
            cudaMalloc(&outputDevice_, bytes);
            outputHost_.resize(elemCount);
        }

        context_->setTensorAddress(name, mode == nvinfer1::TensorIOMode::kINPUT ? inputDevice_ : outputDevice_);
    }
}

std::vector<Detection> TRTDetector::detect(const cv::Mat& frame) {
    // 1. preprocess: letterbox resize + BGR->RGB + normalize + HWC->CHW
    cv::Mat blob = preprocess(frame, inputSize_);   // see preprocessor.hpp

    // 2. copy to device, enqueue, copy back
    cudaMemcpyAsync(inputDevice_, blob.ptr<float>(), blob.total() * sizeof(float),
                     cudaMemcpyHostToDevice, stream_);
    context_->enqueueV3(stream_);   // TensorRT 10.x API; use enqueueV2 on older versions
    cudaMemcpyAsync(outputHost_.data(), outputDevice_, outputHost_.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 3. postprocess: decode boxes, filter by confThresh_, NMS, rescale to original frame
    return postprocess(outputHost_, inputSize_, confThresh_, nmsThresh_);
}

TRTDetector::~TRTDetector() {
    cudaFree(inputDevice_);
    cudaFree(outputDevice_);
    cudaStreamDestroy(stream_);
}