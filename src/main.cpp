#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <utils/config.hpp>

#include "models/rfdetr_model.hpp"

int main(int argc, char** argv) {
    std::string configPath = "config/default.yaml";
    if (argc > 1) {
        configPath = argv[1];
    } else {
        std::cerr << "No config path specified. Falling back to " << configPath << "\n";
    }

    DetectorConfig config;

    try {
        config = loadConfig(configPath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n";
        return 1;
    }

    std::unique_ptr<RFDETRModel> detector;
    try {
        detector = std::make_unique<RFDETRModel>(config);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize RF-DETR Model: " << e.what() << "\n";
        return 1;
    }

    cv::VideoCapture source(config.input_source);
    cv::Mat frame;
    const std::vector<std::string>& classNames = detector->classNames();
    std::vector<Detection> detections;

    while (source.isOpened()) {
        bool ret = source.read(frame);
        if (!ret) {
            // Reset frame position to the very beginning (frame 0)
            source.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;  // Skip to the next iteration to read the first frame again
        }
        try {
            detections = detector->detect(frame);
        } catch (const std::exception& e) {
            std::cerr << "Detection failed: " << e.what() << "\n";
            return 1;
        }
        if (config.enable_profiling) detector->metrics().print(std::cout);
        for (const auto& d : detections) {
            cv::Rect2f box(cv::Point2f(d.box.x1, d.box.y1), cv::Point2f(d.box.x2, d.box.y2));
            cv::rectangle(frame, box, {0, 255, 0}, 2);
            cv::putText(frame, classNames[d.class_id] + " " + std::to_string(d.confidence),
                        box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }
        cv::imshow("detections", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}