#include "models/yolov8_model.hpp"
#include "utils/timer.hpp"

YOLOv8Model::YOLOv8Model(const DetectorConfig& cfg)
    : confThresh_(cfg.conf_threshold), 
    nmsThresh_(cfg.nms_threshold), 
    modelInputWidth_(cfg.input_width), 
    modelInputHeight_(cfg.input_height), 
    sourceWidth_(cfg.source_width), 
    sourceHeight_(cfg.source_height), 
    numClasses_(cfg.num_classes) {
        trt_infer_ = std::make_unique<TRTInfer>(cfg.engine_path);
    }

cv::Mat YOLOv8Model::preprocess_(const cv::Mat& frame) {
    ScopedTimer timer("Preprocess");
    cv::Mat blob;
    cv::dnn::blobFromImage(
        frame, blob,
        1.0 / 255.0,           // scale factor -> normalizes uint8 [0,255] to float [0,1]
        cv::Size(modelInputWidth_, modelInputHeight_),            // resize to this (e.g. 640x640)
        cv::Scalar(0, 0, 0),   // mean subtraction (YOLO models: none)
        true,                  // swapRB -> BGR to RGB
        false                  // crop
    );
    return blob; // shape: [1, 3, H, W] — already NCHW, ready to feed the engine
}

std::vector<Detection> YOLOv8Model::postprocess_(const std::vector<float>& rawOutput) {
    ScopedTimer timer("Postprocess");
    //const int numAttrs = 4 + static_cast<int>(classNames.size()); // 84 for COCO
    const int numAttrs = 4 + numClasses_;
    const int numAnchors = numAttrs * 100;

    // rawOutput is laid out as [1, numAttrs, numAnchors] — i.e. attribute-major,
    // not anchor-major. So element (attr, anchor) lives at rawOutput[attr * numAnchors + anchor].
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    float widthScaleRatio = static_cast<float>(sourceWidth_)/modelInputWidth_;
    float heightScaleRatio = static_cast<float>(sourceHeight_)/modelInputHeight_;

    for (int a = 0; a < numAnchors; ++a) {
        // Find best class score for this anchor
        float bestScore = 0.f;
        int bestClass = -1;
        for (int c = 0; c < 80; ++c) {
            float score = rawOutput[(4 + c) * numAnchors + a];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        
        if (bestScore < confThresh_) continue;

        float cx = rawOutput[0 * numAnchors + a];
        float cy = rawOutput[1 * numAnchors + a];
        float w  = rawOutput[2 * numAnchors + a];
        float h  = rawOutput[3 * numAnchors + a];

        float x1 = (cx - w / 2.f) * widthScaleRatio;
        float y1 = (cy - h / 2.f) * heightScaleRatio;
        float x2 = (cx + w / 2.f) * widthScaleRatio;
        float y2 = (cy + h / 2.f) * heightScaleRatio;

        // Clamp to image bounds
        x1 = std::clamp(x1, 0.f, static_cast<float>(sourceWidth_));
        y1 = std::clamp(y1, 0.f, static_cast<float>(sourceHeight_));
        x2 = std::clamp(x2, 0.f, static_cast<float>(sourceWidth_));
        y2 = std::clamp(y2, 0.f, static_cast<float>(sourceHeight_));

        boxes.emplace_back(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    // Class-aware NMS: cv::dnn::NMSBoxes doesn't separate by class on its own,
    // so either loop per class, or use NMSBoxesBatched with classIds.
    std::vector<int> keepIndices;
    cv::dnn::NMSBoxesBatched(boxes, scores, classIds, confThresh_, nmsThresh_, keepIndices);

    std::vector<Detection> detections;
    detections.reserve(keepIndices.size());
    for (int idx : keepIndices) {
        Detection d;
        d.box = boxes[idx];
        d.confidence = scores[idx];
        d.class_id = classIds[idx];
        //d.class_name = classNames[classIds[idx]];
        d.class_name = "";
        detections.push_back(d);
    }

    return detections;
}

std::vector<float> YOLOv8Model::infer_(const cv::Mat& frame){
    ScopedTimer timer("Inference");
    return trt_infer_->infer(frame);
}