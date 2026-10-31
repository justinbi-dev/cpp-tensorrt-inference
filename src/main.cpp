#include "models/rfdetr_model.hpp"
#include "utils/timer.hpp"
#include <opencv2/opencv.hpp>
#include <utils/config.hpp>

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

    config.engine_path = resolvePath(std::string(PROJECT_ROOT_DIR), config.engine_path);
    auto detector = std::make_unique<RFDETRModel>(config);

    cv::VideoCapture source(resolvePath(std::string(PROJECT_ROOT_DIR), config.input_source));
    cv::Mat frame;

    
    while (source.isOpened()) {
        bool ret = source.read(frame);
        if (!ret) {
            // Reset frame position to the very beginning (frame 0)
            source.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue; // Skip to the next iteration to read the first frame again
        }
        auto detections = detector->detect(frame);
        for (const auto& d : detections) {
            cv::Rect2f box(cv::Point2f(d.box.x1, d.box.y1), cv::Point2f(d.box.x2, d.box.y2));
            cv::rectangle(frame, box, {0, 255, 0}, 2);
            cv::putText(frame, d.class_name + " " + std::to_string(d.confidence),
                        box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }
        cv::imshow("detections", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}