#pragma once
#include <filesystem>
#include <string>

/// Configuration for a single detector run, loaded from a YAML file via
/// loadConfig(). Relative paths in the file are resolved against the config file's directory.
struct DetectorConfig {
    std::string engine_path;       ///< Absolute path to TensorRT engine file.
    float conf_threshold = 0.25f;  ///< Detection confidence threshold, [0, 1].
    std::string input_source;      ///< Absolute path to input source.
    bool enable_profiling = true;  ///< If true, displays per-stage timing on the command line.
};

/// Loads and validates a detector configuration from a YAML file.
///
/// @param path Path to the YAML config file. Relative paths are resolved against the current
/// working directory.
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
/// @throws std::runtime_error if the input path is empty or the resolved path doesn't exist.
std::string resolvePath(const std::filesystem::path& baseDir, const std::string& inputPath);
