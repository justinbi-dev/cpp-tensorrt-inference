#pragma once
#include <string>
#include <vector>

/// Loads and validates a class mapping configuration from a YAML file.
///
/// @param path Path to the YAML class mapping file. Relative paths are resolved against
/// the current working directory.
/// @param numClasses Number of classes in the RF-DETR engine file.
/// @return Populated, validated vector of logit-indexed class names. Unmapped classes are referred
/// to as 'class_<id>'.
/// @throws std::runtime_error if the file is missing, malformed, or fails validation
std::vector<std::string> loadClassNames(const std::string& path, int numClasses);
