#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Detection {
    cv::Rect2f box;      // xyxy or xywh, pick one convention and stick to it
    float confidence;
    int class_id;
    std::string class_name;
};