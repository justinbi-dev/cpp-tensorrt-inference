#include "infer/trt_infer.hpp"

#include <fstream>
#include <cuda_runtime_api.h>

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};
Logger gLogger;

size_t elementSize(nvinfer1::DataType dtype) {
    using nvinfer1::DataType;
    switch (dtype) {
        case DataType::kFLOAT: return 4;
        case DataType::kINT64: return 8;
        case DataType::kINT32: return 4;
        case DataType::kBF16:  return 2;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kUINT8: return 1;
        case DataType::kBOOL:  return 1;
        case DataType::kFP8:   return 1;
        case DataType::kINT4:  break;
    }
    throw std::runtime_error("Unsupported TensorRT data type: " +
                             std::to_string(static_cast<int>(dtype)));
}

std::string nvinferTypeToString(nvinfer1::DataType dtype) {
    using nvinfer1::DataType;
    switch (dtype) {
        case DataType::kFLOAT: return "kFLOAT (32-bit Float)";
        case DataType::kINT64: return "kINT64 (64-bit Int)";
        case DataType::kINT32: return "kINT32 (32-bit Int)";
        case DataType::kBF16:  return "kBF16 (16-bit Brain Float)";
        case DataType::kHALF:  return "kHALF (16-bit Float)";
        case DataType::kINT8:  return "kINT8 (8-bit Quantized Int)";
        case DataType::kUINT8: return "kUINT8 (8-bit Unsigned Int)";
        case DataType::kBOOL:  return "kBOOL (1-bit Boolean)";
        case DataType::kFP8:   return "kFP8 (8-bit Float)";
        case DataType::kINT4:  break;
    }
    throw std::runtime_error("Unsupported TensorRT data type: " +
                             std::to_string(static_cast<int>(dtype)));
}
}  // namespace

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
    const int numIO = engine_->getNbIOTensors();
    tensors_.reserve(numIO);

    for (int i = 0; i < numIO; ++i) {
        TensorBinding tensor;
        TensorInfo& info = tensor.info;
        info.name = engine_->getIOTensorName(i);
        info.is_input = (engine_->getTensorIOMode(info.name.c_str()) == nvinfer1::TensorIOMode::kINPUT);
        info.dtype = engine_->getTensorDataType(info.name.c_str());
        nvinfer1::Dims dims = engine_->getTensorShape(info.name.c_str());

        size_t num_tensor_elements = 1;
        
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] < 0)
                throw std::runtime_error("Dynamic shape detected in tensor binding in " + info.name + " at dimension " + std::to_string(d));
            info.shape.push_back(dims.d[d]);
            num_tensor_elements *= dims.d[d];
        }

        info.bytes = num_tensor_elements * elementSize(info.dtype);

        if (cudaMalloc(&tensor.device_buffer, info.bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed for tensor " + info.name);
        context_->setTensorAddress(info.name.c_str(), tensor.device_buffer);

        if (info.is_input) {
            input_indices_.push_back(tensors_.size());
        }
        else {
            if (info.dtype != nvinfer1::DataType::kFLOAT){
                throw std::runtime_error("TRT Infer currently only supports floating point outputs. Tensor " + info.name + " is of type " + nvinferTypeToString(info.dtype));
            }
            output_indices_.push_back(tensors_.size());
            output_.out_host_buffers.emplace(info.name, std::vector<float>(info.bytes / sizeof(float)));
        }
        tensors_.push_back(std::move(tensor));

    }

    if (input_indices_.size() != 1) {
        throw std::runtime_error("TRTInfer currently only supports a single input tensor; engine has " + std::to_string(input_indices_.size()));
    }
}

const InferenceResult& TRTInfer::infer(const cv::Mat& blob) {
    const TensorBinding& in = tensors_[input_indices_[0]];
    const size_t blob_bytes = blob.total() * blob.elemSize();
    if (blob_bytes != in.info.bytes) {
        throw std::runtime_error("Input size mismatch for '" + in.info.name + "': got " +
                                std::to_string(blob_bytes) + " bytes, engine expects " +
                                std::to_string(in.info.bytes));
    }

    cudaEvent_t start, afterH2D, afterInfer, afterD2H;
    cudaEventCreate(&start);
    cudaEventCreate(&afterH2D);
    cudaEventCreate(&afterInfer);
    cudaEventCreate(&afterD2H);

    cudaEventRecord(start, stream_);

    cudaMemcpyAsync(in.device_buffer, blob.ptr<float>(), in.info.bytes, cudaMemcpyHostToDevice, stream_);
    cudaEventRecord(afterH2D, stream_);

    context_->enqueueV3(stream_);
    cudaEventRecord(afterInfer, stream_);

    for (size_t idx : output_indices_) {
        const TensorBinding& out = tensors_[idx];
        cudaMemcpyAsync(output_.out_host_buffers.at(out.info.name).data(), out.device_buffer, out.info.bytes, cudaMemcpyDeviceToHost, stream_);
    }
    cudaEventRecord(afterD2H, stream_);

    cudaStreamSynchronize(stream_); // must sync before reading event times

    float h2d_ms = 0.f, infer_ms = 0.f, d2h_ms = 0.f, total_ms = 0.f;
    cudaEventElapsedTime(&output_.metrics.h2d_ms, start, afterH2D);
    cudaEventElapsedTime(&output_.metrics.infer_ms, afterH2D, afterInfer);
    cudaEventElapsedTime(&output_.metrics.d2h_ms, afterInfer, afterD2H);
    cudaEventElapsedTime(&output_.metrics.total_ms, start, afterD2H);

    cudaEventDestroy(start);
    cudaEventDestroy(afterH2D);
    cudaEventDestroy(afterInfer);
    cudaEventDestroy(afterD2H);

    return output_;
}

const TensorInfo& TRTInfer::inputTensorInfo() const {
    return tensors_[input_indices_[0]].info;
}

const TensorInfo& TRTInfer::outputTensorInfo(const std::string& name) const {
    for (size_t idx : output_indices_) {
        if (tensors_[idx].info.name == name){
            return tensors_[idx].info;
        }
    }
    throw std::runtime_error("Tensor of name " + name + " not found.");
}

TRTInfer::~TRTInfer() {
    for (auto& t : tensors_) cudaFree(t.device_buffer);
    cudaStreamDestroy(stream_);
}