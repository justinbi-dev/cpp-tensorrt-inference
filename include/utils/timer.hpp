#pragma once
#include <chrono>

class ScopedTimer {
public:
    explicit ScopedTimer(float& out_ms) : out_(out_ms), start_(std::chrono::steady_clock::now()) {}

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    
    ~ScopedTimer() {
        out_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    float& out_;
    std::chrono::steady_clock::time_point start_;
};
