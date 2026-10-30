#include "infer/trt_infer.hpp"

#include <fstream>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static Logger gLogger;


TRTInfer::TRTInfer(const std::string& path) {
    loadEngine(path);
    allocateBuffers();
    cudaStreamCreate(&stream_);
}
void TRTInfer::loadEngine(const std::string& path) {
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
void TRTInfer::allocateBuffers() {
    int numIO = engine_->getNbIOTensors();

    for (int i = 0; i < numIO; ++i) {
        const char* name = engine_->getIOTensorName(i);
        bool isInput = (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        nvinfer1::Dims dims = engine_->getTensorShape(name);

        std::cout << "Tensor Binding " << i << ": " << name << (isInput ? "(input tensor) " : "(output tensor) ") << "with shape: [";

        size_t elemCount = 1;
        std::vector<int64_t> shape;

        for (int d = 0; d < dims.nbDims; ++d) {
            std::cout << dims.d[d] << ", ";
            elemCount *= dims.d[d];
            shape.push_back(dims.d[d]);
        }
        std::cout << "]" << std::endl;

        size_t bytes = elemCount * sizeof(float);

        void* device_buffer;
        cudaMalloc(&device_buffer, bytes);
        if (isInput) {
            in_device_buffers_.push_back(device_buffer);
            in_shapes.push_back(shape);

        } else {
            out_device_buffers_.push_back(device_buffer);
            out_shapes.push_back(shape);
            output_.out_host_buffers.emplace_back(elemCount);
        }

        context_->setTensorAddress(name, device_buffer);
    }
}
InferenceResult TRTInfer::infer(const cv::Mat& blob) {
    cudaEvent_t start, afterH2D, afterInfer, afterD2H;
    cudaEventCreate(&start);
    cudaEventCreate(&afterH2D);
    cudaEventCreate(&afterInfer);
    cudaEventCreate(&afterD2H);

    cudaEventRecord(start, stream_);

    cudaMemcpyAsync(in_device_buffers_[0], blob.ptr<float>(), blob.total() * sizeof(float),
                     cudaMemcpyHostToDevice, stream_);
    cudaEventRecord(afterH2D, stream_);

    context_->enqueueV3(stream_);
    cudaEventRecord(afterInfer, stream_);

    for (int i = 0; i < out_device_buffers_.size(); ++i) {
        cudaMemcpyAsync(output_.out_host_buffers[i].data(), out_device_buffers_[i],
                         output_.out_host_buffers[i].size() * sizeof(float),
                         cudaMemcpyDeviceToHost, stream_);
    }
    cudaEventRecord(afterD2H, stream_);

    cudaStreamSynchronize(stream_); // must sync before reading event times

    float h2d_ms = 0.f, infer_ms = 0.f, d2h_ms = 0.f, total_ms = 0.f;
    cudaEventElapsedTime(&output_.metrics.h2d_ms, start, afterH2D);
    cudaEventElapsedTime(&output_.metrics.infer_ms, afterH2D, afterInfer);
    cudaEventElapsedTime(&output_.metrics.d2h_ms, afterInfer, afterD2H);
    cudaEventElapsedTime(&output_.metrics.total_ms, start, afterD2H);

    // std::cout << "\r\033[K[TRTInfer] "
    //         << std::fixed << std::setprecision(3)
    //         << "H2D copy: " << h2d_ms << " ms | "
    //         << "Inference: " << infer_ms << " ms | "
    //         << "D2H copy: " << d2h_ms << " ms | "
    //         << "Total: " << total_ms << " ms"
    //         << std::flush; // Crucial: forces immediate console update

    cudaEventDestroy(start);
    cudaEventDestroy(afterH2D);
    cudaEventDestroy(afterInfer);
    cudaEventDestroy(afterD2H);

    return output_;
}

TRTInfer::~TRTInfer() {
    for (int i = 0; i < in_device_buffers_.size(); i++){
        cudaFree(in_device_buffers_[i]);
    }
    for (int i = 0; i < out_device_buffers_.size(); i++){
        cudaFree(out_device_buffers_[i]);
    }
    cudaStreamDestroy(stream_);
}