#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <stdexcept>

struct InferenceMetrics {
    float h2d_ms = 0.0f;
    float infer_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_ms = 0.0f;
};

struct InferenceResult {
    InferenceMetrics metrics;
    std::unordered_map<std::string, std::vector<float>> out_host_buffers;

    const std::vector<float>& get(const std::string& name) const {
        auto it = out_host_buffers.find(name);
        if (it == out_host_buffers.end())
            throw std::runtime_error("No output tensor named '" + name + "'");
        return it->second;
    }
};