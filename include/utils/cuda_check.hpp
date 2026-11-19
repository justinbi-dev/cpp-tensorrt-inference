#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess)                                                                   \
            throw std::runtime_error(std::string(cudaGetErrorName(err_)) + ": " +                  \
                                     cudaGetErrorString(err_) + " in " #call " at " __FILE__ ":" + \
                                     std::to_string(__LINE__));                                    \
    } while (0)
