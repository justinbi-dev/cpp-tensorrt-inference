#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "types.hpp"

struct LetterboxInfo {
    float scale;   // resize ratio applied
    int padX;      // horizontal padding added
    int padY;      // vertical padding added
};

std::vector<Detection> postprocess(
    const std::vector<float>& rawOutput,   // flat buffer, layout [1, 84, 8400]
    //const LetterboxInfo& lb,
    cv::Size originalSize, 
    float confThresh,
    float nmsThresh
    //const std::vector<std::string>& classNames
);