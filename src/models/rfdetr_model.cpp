#include "models/rfdetr_model.hpp"

RFDETRModel::RFDETRModel(const DetectorConfig& cfg)
    : confThresh_(cfg.conf_threshold) {
        trt_infer_ = std::make_unique<TRTInfer>(cfg.engine_path);
        const TensorInfo& input_info = trt_infer_->inputTensorInfo();

        if (input_info.shape.size() != 4) throw std::runtime_error("Input tensor has: " + std::to_string(input_info.shape.size()) + " dimensions. Expected 4 dimensions.");
        modelInputHeight_ = static_cast<int>(input_info.shape[2]);
        modelInputWidth_ = static_cast<int>(input_info.shape[3]);

        const TensorInfo& label_info = trt_infer_->outputTensorInfo("labels");
        if (label_info.shape.size() != 3) throw std::runtime_error("Output label tensor has: " + std::to_string(label_info.shape.size()) + " dimensions. Expected 3 dimensions.");
        numClasses_ = static_cast<int>(label_info.shape[2]);
        numQueries_ = static_cast<int>(label_info.shape[1]);
    }

cv::Mat RFDETRModel::preprocess_(const cv::Mat& frame) {
    ManualTimer timer;
    timer.start_timer();
    cv::dnn::Image2BlobParams params(
        cv::Scalar(1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225)),  // 1/(255*std)
        cv::Size(modelInputWidth_, modelInputHeight_),
        cv::Scalar(123.675, 116.28, 103.53),                                        // 255*mean
        true);                                                                       // swapRB
    cv::Mat blob = cv::dnn::blobFromImageWithParams(frame, params);

    metrics_.preprocess_ms = timer.get_ms();
    return blob; // shape: [1, 3, H, W] — already NCHW, ready to feed the engine
}

std::vector<Detection> RFDETRModel::postprocess_(const InferenceResult& inference_result, int sourceWidth, int sourceHeight) {
    ManualTimer timer;
    timer.start_timer();
    metrics_.inference_metrics = inference_result.metrics;

    std::vector<Detection> detections;

    const std::vector<float>& labels = inference_result.get("labels");
    const std::vector<float>& dets   = inference_result.get("dets");


    for (int i = 0; i < numQueries_; ++i) {
        float bestScore = 0.f;
        int bestClass = -1;
        for (int c = 0; c < numClasses_; ++c) {
            float score = 1.0 / (1.0 + std::exp(-labels[(numClasses_*i) + c]));
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        if (bestScore < confThresh_) continue;

        float cx = dets[(4*i)];
        float cy = dets[(4*i)+1];
        float w = dets[(4*i)+2];
        float h = dets[(4*i)+3];



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
        //d.class_name = classNames[classIds[idx]];
        d.class_name = "";
        detections.push_back(d);
    }
    metrics_.postprocess_ms = timer.get_ms();

    return detections;
}

InferenceResult RFDETRModel::infer_(const cv::Mat& frame){
    ManualTimer timer;
    timer.start_timer();
    InferenceResult result = trt_infer_->infer(frame);
    metrics_.infer_ms = timer.get_ms();
    return result;
}

std::vector<Detection> RFDETRModel::detect(const cv::Mat& frame) {
    ManualTimer timer;
    timer.start_timer();
    cv::Mat input = preprocess_(frame);
    InferenceResult raw_output = infer_(input);

    std::vector<Detection> detections = postprocess_(raw_output, frame.cols, frame.rows);
    metrics_.total_ms = timer.get_ms();
    return detections;
}