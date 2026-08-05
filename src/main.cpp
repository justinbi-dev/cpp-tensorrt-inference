#include "trt_detector.hpp"
#include "video_source.hpp"
#include <opencv2/opencv.hpp>
#include <preprocessor.hpp>
#include <postprocessor.hpp>
#include <config.hpp>

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "."
#endif

int main(int argc, char** argv) {

    std::string configPath = resolvePath(std::string(PROJECT_ROOT_DIR), "config/default.yaml");
    if (argc > 1) {
        configPath = resolvePath(std::string(PROJECT_ROOT_DIR), argv[1]);
    }
    else {
        std::cerr << "No config path specified. Falling back to default path. \n";
    }

    DetectorConfig config;
    try {
        config = loadConfig(configPath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n";
        return 1;
    }

    auto detector = std::make_unique<TRTDetector>(resolvePath(std::string(PROJECT_ROOT_DIR), config.engine_path));

    cv::VideoCapture source(resolvePath(std::string(PROJECT_ROOT_DIR), config.input_source));
    cv::Mat frame;

    while (source.isOpened()) {
        source >> frame;
        auto detections = detector->detect(frame);
        for (const auto& d : detections) {
            cv::rectangle(frame, d.box, {0, 255, 0}, 2);
            cv::putText(frame, d.class_name + " " + std::to_string(d.confidence),
                        d.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }
        cv::imshow("detections", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}