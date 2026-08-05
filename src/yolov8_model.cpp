#include "yolov8_model.hpp"

YOLOv8Model::YOLOv8Model(const std::string& enginePath,
                        float confThresh = 0.25f,
                        float nmsThresh = 0.45f,
                        uint modelInputWidth = 640,
                        uint modelInputHeight = 640,
                        uint sourceWidth = 1920,
                        uint sourceHeight = 1080)
    : confThresh_(confThresh), nmsThresh_(nmsThresh), modelInputWidth_(modelInputWidth), modelInputHeight_(modelInputHeight), sourceWidth_(sourceWidth), sourceHeight_(sourceHeight) {}

std::vector<float> YOLOv8Model::preprocess(const cv::Mat& frame) {
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

std::vector<Detection> postprocess(const std::vector<float>& rawOutput);
std::vector<float> YOLOv8Model::preprocess(const cv::Mat& frame) {
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