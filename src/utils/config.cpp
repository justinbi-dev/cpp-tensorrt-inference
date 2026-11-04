#include <yaml-cpp/yaml.h>
#include "utils/config.hpp"
#include <stdexcept>
#include <cmath>

std::string resolvePath(const std::filesystem::path& baseDir,  const std::string& inputPath) {
    if (inputPath.empty()) {
        throw std::runtime_error("Input path is empty.");
    } 
    
    std::filesystem::path p(inputPath);

    bool resolved = p.is_absolute();
    
    if (!resolved) { 
        p = (baseDir / p).lexically_normal();
    }

    if (!std::filesystem::exists(p)) {
        throw std::runtime_error("Path does not exist: " + p.string());
    }
    return p.string();
}

DetectorConfig loadConfig(const std::string& path) {
    YAML::Node node = YAML::LoadFile(path); // throws YAML::BadFile if missing

    const std::filesystem::path configDir = std::filesystem::absolute(path).parent_path();

    if (!node["engine_path"]) throw std::runtime_error("Missing required key 'engine_path' in " + path);
    if (!node["input_source"]) throw std::runtime_error("Missing required key 'input_source' in " + path);

    DetectorConfig cfg;
    cfg.engine_path       = resolvePath(configDir, node["engine_path"].as<std::string>());
    cfg.conf_threshold    = node["conf_threshold"].as<float>(cfg.conf_threshold);
    cfg.input_source      = resolvePath(configDir, node["input_source"].as<std::string>());
    cfg.enable_profiling  = node["enable_profiling"].as<bool>(cfg.enable_profiling);

    if (std::isnan(cfg.conf_threshold) || cfg.conf_threshold < 0.f || cfg.conf_threshold > 1.f) {
        throw std::runtime_error("conf_threshold must be within range [0, 1]");
    }

    return cfg;
}
