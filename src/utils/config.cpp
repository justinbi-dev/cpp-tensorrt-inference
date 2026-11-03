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
        throw std::runtime_error("Config path does not exist: " + p.string());
    }
    return p.string();
}

DetectorConfig loadConfig(const std::string& path) {
    YAML::Node node = YAML::LoadFile(path); // throws YAML::BadFile if missing

    DetectorConfig cfg;
    cfg.engine_path       = node["engine_path"].as<std::string>();
    cfg.conf_threshold    = node["conf_threshold"].as<float>(cfg.conf_threshold);
    cfg.input_source      = node["input_source"].as<std::string>();
    cfg.enable_profiling  = node["enable_profiling"].as<bool>(cfg.enable_profiling);

    if (cfg.engine_path.empty()) {
        throw std::runtime_error("engine_path must be specified.");
    }
    if (std::isnan(cfg.conf_threshold) || cfg.conf_threshold < 0.f || cfg.conf_threshold > 1.f) {
        throw std::runtime_error("conf_threshold must be within range [0, 1]");
    }
    if (cfg.input_source.empty()) {
        throw std::runtime_error("input_source must be specified.");
    }
    return cfg;
}
