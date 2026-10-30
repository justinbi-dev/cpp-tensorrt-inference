#pragma once
#include <vector>

struct InferenceMetrics {
    float h2d_ms = 0.0f;
    float infer_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_ms = 0.0f;
};

struct InferenceResult {
    InferenceMetrics metrics;
    std::vector<std::vector<float>> out_host_buffers;
};