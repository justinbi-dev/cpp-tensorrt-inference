#pragma once
#include <iostream>
#include <chrono>
#include <string>

class ScopedTimer {
public:
    ScopedTimer(std::string name) : name_(std::move(name)), start_(std::chrono::high_resolution_clock::now()) {}
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << "[" << name_ << "] took " << duration << " ms\n";
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};


struct ManualTimer {
    std::chrono::high_resolution_clock::time_point start;
    
    void start_timer() { start = std::chrono::high_resolution_clock::now(); }
    
    float get_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        using ms = std::chrono::duration<float, std::milli>;
        return std::chrono::duration_cast<ms>(std::chrono::high_resolution_clock::now() - start).count();
    }
};