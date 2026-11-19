#include "models/rfdetr_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "utils/class_names.hpp"
#include "utils/timer.hpp"

RFDETRModel::RFDETRModel(const DetectorConfig& cfg) : confThresh_(cfg.conf_threshold) {
    trt_infer_ = std::make_unique<TRTInfer>(cfg.engine_path);
    const TensorInfo& input_info = trt_infer_->inputTensorInfo();

    if (input_info.shape.size() != 4)
        throw std::runtime_error("Input tensor has: " + std::to_string(input_info.shape.size()) +
                                 " dimensions. Expected 4 dimensions.");
    if (input_info.shape[0] != 1)
        throw std::runtime_error(
            "Input tensor has a batch size of: " + std::to_string(input_info.shape[0]) +
            ". Only single-image batches are currently supported.");
    if (input_info.shape[1] != 3)
        throw std::runtime_error("Input tensor has: " + std::to_string(input_info.shape[1]) +
                                 " input channels. Expected 3 channels.");
    modelInputHeight_ = static_cast<int>(input_info.shape[2]);
    modelInputWidth_ = static_cast<int>(input_info.shape[3]);

    const TensorInfo& label_info = trt_infer_->outputTensorInfo("labels");
    if (label_info.shape.size() != 3)
        throw std::runtime_error(
            "Output label tensor has: " + std::to_string(label_info.shape.size()) +
            " dimensions. Expected 3 dimensions.");
    if (label_info.shape[0] != 1)
        throw std::runtime_error(
            "Output label tensor has a batch size of: " + std::to_string(label_info.shape[0]) +
            ". Only single-image batches are currently supported.");
    numClasses_ = static_cast<int>(label_info.shape[2]);
    numQueries_ = static_cast<int>(label_info.shape[1]);

    const TensorInfo& det_info = trt_infer_->outputTensorInfo("dets");
    if (det_info.shape.size() != 3)
        throw std::runtime_error(
            "Output detection tensor has: " + std::to_string(det_info.shape.size()) +
            " dimensions. Expected 3 dimensions.");
    if (det_info.shape[0] != 1)
        throw std::runtime_error(
            "Output detection tensor has a batch size of: " + std::to_string(det_info.shape[0]) +
            ". Only single-image batches are currently supported.");
    if (det_info.shape[1] != label_info.shape[1])
        throw std::runtime_error("Output tensor number of queries does not match: dets = " +
                                 std::to_string(det_info.shape[1]) +
                                 ", labels = " + std::to_string(label_info.shape[1]));
    if (det_info.shape[2] != 4)
        throw std::runtime_error("Output detection tensor has " +
                                 std::to_string(det_info.shape[2]) +
                                 " box coordinates. Expected 4 coordinates.");

    classNames_ = loadClassNames(cfg.class_map_path, numClasses_);
}

cv::Mat RFDETRModel::preprocess_(const cv::Mat& frame) {
    ScopedTimer timer(metrics_.preprocess_ms);

    cv::dnn::Image2BlobParams params(
        cv::Scalar(1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225)),  // 1/(255*std)
        cv::Size(modelInputWidth_, modelInputHeight_),
        cv::Scalar(123.675, 116.28, 103.53),  // 255*mean
        true);                                // swapRB
    cv::Mat blob = cv::dnn::blobFromImageWithParams(frame, params);

    return blob;  // shape: [1, 3, H, W] — already NCHW, ready to feed the engine
}

std::vector<Detection> RFDETRModel::postprocess_(const InferenceResult& inference_result,
                                                 int sourceWidth, int sourceHeight) {
    ScopedTimer timer(metrics_.postprocess_ms);

    metrics_.inference_metrics = inference_result.metrics;

    std::vector<Detection> detections;

    const std::vector<float>& labels = inference_result.get("labels");
    const std::vector<float>& dets = inference_result.get("dets");

    for (int i = 0; i < numQueries_; ++i) {
        float bestScore = 0.f;
        int bestClass = -1;
        for (int c = 0; c < numClasses_; ++c) {
            float score = 1.0 / (1.0 + std::exp(-labels[(numClasses_ * i) + c]));
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        if (bestScore < confThresh_) continue;

        float cx = dets[(4 * i)];
        float cy = dets[(4 * i) + 1];
        float w = dets[(4 * i) + 2];
        float h = dets[(4 * i) + 3];

        float x1 = (cx - w / 2.f) * static_cast<float>(sourceWidth);
        float y1 = (cy - h / 2.f) * static_cast<float>(sourceHeight);
        float x2 = (cx + w / 2.f) * static_cast<float>(sourceWidth);
        float y2 = (cy + h / 2.f) * static_cast<float>(sourceHeight);
        // Clamp to image bounds
        x1 = std::clamp(x1, 0.f, static_cast<float>(sourceWidth));
        y1 = std::clamp(y1, 0.f, static_cast<float>(sourceHeight));
        x2 = std::clamp(x2, 0.f, static_cast<float>(sourceWidth));
        y2 = std::clamp(y2, 0.f, static_cast<float>(sourceHeight));

        Detection d;
        d.box = {x1, y1, x2, y2};
        d.confidence = bestScore;
        d.class_id = bestClass;
        detections.push_back(d);
    }

    return detections;
}

const InferenceResult& RFDETRModel::infer_(const cv::Mat& frame) {
    ScopedTimer timer(metrics_.infer_ms);

    return trt_infer_->infer(frame);
}

std::vector<Detection> RFDETRModel::detect(const cv::Mat& frame) {
    ScopedTimer timer(metrics_.total_ms);
    cv::Mat input = preprocess_(frame);
    const InferenceResult& raw_output = infer_(input);

    return postprocess_(raw_output, frame.cols, frame.rows);
}