// Usage: orin_benchmark <config.yaml> <num_frames> <warmup_frames> <out.csv>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/rfdetr_model.hpp"
#include "utils/config.hpp"

namespace {

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " <config.yaml> <num_frames> <warmup_frames> <out.csv>\n"
              << "  num_frames     Frames to record after warm-up (> 0)\n"
              << "  warmup_frames  Frames to run before recording, not written to the CSV (>= 0)\n";
}

int parseInt(const std::string& text, const std::string& name) {
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + " must be an integer, got '" + text + "'");
    }
    if (parsed != text.size())
        throw std::invalid_argument(name + " must be an integer, got '" + text + "'");
    return value;
}

// Wall-clock columns come from ScopedTimer (CPU side, includes waiting on the GPU). The gpu_*
// columns come from CUDA events recorded on the inference stream inside TRTInfer.
void writeHeader(std::ostream& os) {
    os << "frame,total_ms,preprocess_ms,infer_ms,postprocess_ms,"
          "gpu_total_ms,gpu_h2d_ms,gpu_compute_ms,gpu_d2h_ms,num_detections\n";
}

void writeRow(std::ostream& os, int frame, const PerformanceMetrics& m, size_t numDetections) {
    const InferenceMetrics& gpu = m.inference_metrics;
    os << frame << ',' << m.total_ms << ',' << m.preprocess_ms << ',' << m.infer_ms << ','
       << m.postprocess_ms << ',' << gpu.total_ms << ',' << gpu.h2d_ms << ',' << gpu.infer_ms << ','
       << gpu.d2h_ms << ',' << numDetections << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string configPath = argv[1];
    const std::string csvPath = argv[4];

    int numFrames = 0;
    int warmupFrames = 0;
    try {
        numFrames = parseInt(argv[2], "num_frames");
        warmupFrames = parseInt(argv[3], "warmup_frames");
    } catch (const std::exception& e) {
        std::cerr << "Invalid argument: " << e.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }
    if (numFrames <= 0 || warmupFrames < 0) {
        std::cerr << "Invalid argument: num_frames must be > 0 and warmup_frames must be >= 0\n";
        printUsage(argv[0]);
        return 1;
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
    if (!source.isOpened()) {
        std::cerr << "Failed to open input source: " << config.input_source << "\n";
        return 1;
    }

    std::ofstream csv(csvPath);
    if (!csv) {
        std::cerr << "Failed to open output file: " << csvPath << "\n";
        return 1;
    }
    writeHeader(csv);

    std::cout << "Engine: " << config.engine_path << "\n"
              << "Input: " << config.input_source << "\n"
              << "Warm-up frames: " << warmupFrames << ", measured frames: " << numFrames << "\n";

    cv::Mat frame;
    const int totalFrames = warmupFrames + numFrames;
    try {
        for (int i = 0; i < totalFrames; ++i) {
            if (!source.read(frame)) {
                // End of the video: rewind and try once more so short clips can be looped. A
                // second failure means the source is unreadable, not just finished.
                source.set(cv::CAP_PROP_POS_FRAMES, 0);
                if (!source.read(frame))
                    throw std::runtime_error("Failed to read a frame from " + config.input_source);
            }

            const std::vector<Detection> detections = detector->detect(frame);

            if (i >= warmupFrames)
                writeRow(csv, i - warmupFrames, detector->metrics(), detections.size());
        }
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Wrote " << numFrames << " frames to " << csvPath << "\n";
    return 0;
}
