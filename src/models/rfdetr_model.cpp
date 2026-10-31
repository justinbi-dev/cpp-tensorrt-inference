#include "models/rfdetr_model.hpp"

RFDETRModel::RFDETRModel(const DetectorConfig& cfg)
    : confThresh_(cfg.conf_threshold), 
    modelInputWidth_(cfg.input_width), 
    modelInputHeight_(cfg.input_height), 
    sourceWidth_(cfg.source_width), 
    sourceHeight_(cfg.source_height), 
    numClasses_(cfg.num_classes) {
        trt_infer_ = std::make_unique<TRTInfer>(cfg.engine_path);
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

    metrics.preprocess_ms = timer.get_ms();
    return blob; // shape: [1, 3, H, W] — already NCHW, ready to feed the engine
}

std::vector<Detection> RFDETRModel::postprocess_(InferenceResult& inference_result) {
    ManualTimer timer;
    timer.start_timer();
    metrics.inference_metrics = inference_result.metrics;

    float widthScaleRatio = static_cast<float>(sourceWidth_);
    float heightScaleRatio = static_cast<float>(sourceHeight_);
    
    detections.clear();

    const std::vector<float>& labels = inference_result.get("labels");
    const std::vector<float>& dets   = inference_result.get("dets");
    //std::transform(labels.begin(), labels.end(), labels.begin(), [](double x) {return 1.0 / (1.0 + std::exp(-x));});

    for (int i = 0; i < 300; ++i) {
        float bestScore = 0.f;
        int bestClass = -1;
        for (int c = 0; c < numClasses_; ++c) {
            float score = 1.0 / (1.0 + std::exp(-labels[(numClasses_*i) + c]));
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        //std::cout << bestScore << std::endl;
        if (bestScore < confThresh_) continue;

        float cx = dets[(4*i)];
        float cy = dets[(4*i)+1];
        float w = dets[(4*i)+2];
        float h = dets[(4*i)+3];



        float x1 = (cx - w / 2.f) * widthScaleRatio;
        float y1 = (cy - h / 2.f) * heightScaleRatio;
        float x2 = (cx + w / 2.f) * widthScaleRatio;
        float y2 = (cy + h / 2.f) * heightScaleRatio;
        // Clamp to image bounds
        x1 = std::clamp(x1, 0.f, static_cast<float>(sourceWidth_));
        y1 = std::clamp(y1, 0.f, static_cast<float>(sourceHeight_));
        x2 = std::clamp(x2, 0.f, static_cast<float>(sourceWidth_));
        y2 = std::clamp(y2, 0.f, static_cast<float>(sourceHeight_));

        
        Detection d;
        d.box = {x1, y1, x2, y2};
        d.confidence = bestScore;
        d.class_id = bestClass;
        //d.class_name = classNames[classIds[idx]];
        d.class_name = "";
        detections.push_back(d);
    }

    metrics.postprocess_ms = timer.get_ms();

    return detections;
}

InferenceResult RFDETRModel::infer_(const cv::Mat& frame){
    ManualTimer timer;
    timer.start_timer();
    InferenceResult result = trt_infer_->infer(frame);
    metrics.infer_ms = timer.get_ms();
    return result;
}

std::vector<Detection> RFDETRModel::detect(const cv::Mat& frame) {
    ManualTimer timer;
    timer.start_timer();
    cv::Mat input = preprocess_(frame);
    InferenceResult raw_output = infer_(input);
    postprocess_(raw_output);
    metrics.total_ms = timer.get_ms();
    metrics.print(std::cout);
    return detections;
}