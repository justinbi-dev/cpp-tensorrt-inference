#include "models/performance_metrics.hpp"

#include <iomanip>
#include <iostream>

void PerformanceMetrics::print() const {
    print(std::cout);
}

void PerformanceMetrics::print(std::ostream& os) const {
    // Save and restore stream state so the caller's formatting isn't changed
    const auto flags = os.flags();
    const auto prec = os.precision();

    os << "\r\033[K" << std::fixed << std::setprecision(3)
       << "Total: " << total_ms << "ms | "
       << "Preprocess: " << preprocess_ms << " ms | "
       << "Inference: " << infer_ms << " ms | "
       << "[" 
       << "Inference total: " << inference_metrics.total_ms << " ms | "
       << "H2D copy: " << inference_metrics.h2d_ms << " ms | "
       << "NVInfer: " << inference_metrics.infer_ms << " ms | "
       << "D2H copy: " << inference_metrics.d2h_ms << " ms"
       << "] | "
       << "Postprocess: " << postprocess_ms << "ms"
       << std::flush;


    os.flags(flags);
    os.precision(prec);
}