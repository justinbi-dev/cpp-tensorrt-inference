#include <yaml-cpp/yaml.h>
#include "utils/config.hpp"
#include <stdexcept>

std::string resolvePath(const std::filesystem::path& baseDir,  const std::string& inputPath) {
    if (inputPath.empty()){
        return inputPath;
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
    cfg.engine_path      = node["engine_path"].as<std::string>();
    cfg.conf_threshold    = node["conf_threshold"].as<float>(0.5f);
    cfg.nms_threshold      = node["nms_threshold"].as<float>(0.45f);
    cfg.input_width        = node["input_width"].as<int>(640);
    cfg.input_height       = node["input_height"].as<int>(640);
    cfg.source_width        = node["source_width"].as<int>(1920);
    cfg.source_height       = node["source_height"].as<int>(1080);
    cfg.num_classes       = node["num_classes"].as<int>(80);
    cfg.input_source      = node["input_source"].as<std::string>("camera");
    cfg.enable_profiling  = node["enable_profiling"].as<bool>(false);

    if (cfg.engine_path.empty()) {
        throw std::runtime_error("Config must specify engine_path or onnx_path");
    }
    return cfg;
}