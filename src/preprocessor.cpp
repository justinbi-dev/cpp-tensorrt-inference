#include "preprocessor.hpp"

cv::Mat preprocess(const cv::Mat& frame, cv::Size targetSize) {
    cv::Mat blob;
    cv::dnn::blobFromImage(
        frame, blob,
        1.0 / 255.0,           // scale factor -> normalizes uint8 [0,255] to float [0,1]
        targetSize,            // resize to this (e.g. 640x640)
        cv::Scalar(0, 0, 0),   // mean subtraction (YOLO models: none)
        true,                  // swapRB -> BGR to RGB
        false                  // crop
    );
    return blob; // shape: [1, 3, H, W] — already NCHW, ready to feed the engine
}