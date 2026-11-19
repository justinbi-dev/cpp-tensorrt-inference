cmake_minimum_required(VERSION 3.16)
project(orin_detector LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT isn't always found via a CMake config on Jetson — point at it directly.
set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)
link_directories(${TENSORRT_LIB_DIR})

add_library(detector_core
    src/infer/trt_infer.cpp
    src/utils/config.cpp
    src/utils/class_names.cpp
    src/models/performance_metrics.cpp
    src/models/rfdetr_model.cpp
)

target_link_libraries(detector_core PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    yaml-cpp
)

add_executable(orin_detector src/main.cpp)
target_link_libraries(orin_detector PRIVATE detector_core)

