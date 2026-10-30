#pragma once
#include <iostream>
#include "infer/inference_result.hpp"

struct PerformanceMetrics {
    float total_ms = 0.0f;
    float preprocess_ms = 0.0f;
    float infer_ms = 0.0f;
    float postprocess_ms = 0.0f;
    InferenceMetrics inference_metrics;

    
    void print() const;
    void print(std::ostream& os) const;
};