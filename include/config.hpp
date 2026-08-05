#pragma once
#include <string>
#include <filesystem>

/// Configuration for a single detector run, loaded from a YAML file via
/// loadConfig(). All path fields are absolute after loading (see
/// loadConfig() for resolution rules).
struct DetectorConfig {
    std::string engine_path;             ///< Absolute path to TensorRT engine file.
    float conf_threshold = 0.5f;         ///< Detection confidence threshold, [0, 1].
    float nms_threshold = 0.45f;         ///< IoU threshold for NMS suppression, [0, 1].
    int input_width = 640;               ///< Model input width, in pixels.
    int input_height = 640;              ///< Model input height, in pixels.
    std::string input_source;            ///< "camera" or a path to a video/image file.
    bool enable_profiling = false;       ///< If true, records per-stage timing.
};

/// Loads and validates a detector configuration from a YAML file.
///
/// Path fields (e.g. engine_path) are resolved relative to the config
/// file's own directory via resolvePath(), so the returned struct's
/// paths are absolute regardless of the caller's working directory.
///
/// @param path Path to the YAML config file (resolved relative to cwd).
/// @return Populated, validated DetectorConfig.
/// @throws std::runtime_error if the file is missing, malformed, fails
///         validation, or references a path that doesn't exist on disk.
DetectorConfig loadConfig(const std::string& path);

/// Resolves `inputPath` against `baseDir` if it is not already absolute,
/// normalizing the result (e.g. collapsing ".."), and verifies it exists.
///
/// @param baseDir Directory to resolve `inputPath` against, if relative.
/// @param inputPath A relative or absolute filesystem path.
/// @return The resolved, absolute path as a string.
/// @throws std::runtime_error if the resolved path does not exist.
std::string resolvePath(const std::filesystem::path& baseDir, const std::string& inputPath);