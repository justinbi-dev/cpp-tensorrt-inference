#include "postprocessor.hpp"

std::vector<Detection> postprocess(
    const std::vector<float>& rawOutput,
    //const LetterboxInfo& lb,
    cv::Size originalSize,
    float confThresh,
    float nmsThresh
    //const std::vector<std::string>& classNames
) {
    //const int numAttrs = 4 + static_cast<int>(classNames.size()); // 84 for COCO
    const int numAttrs = 84;
    const int numAnchors = 8400;

    // rawOutput is laid out as [1, numAttrs, numAnchors] — i.e. attribute-major,
    // not anchor-major. So element (attr, anchor) lives at rawOutput[attr * numAnchors + anchor].
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for (int a = 0; a < numAnchors; ++a) {
        // Find best class score for this anchor
        float bestScore = 0.f;
        int bestClass = -1;
        for (int c = 0; c < 80; ++c) {
            float score = rawOutput[(4 + c) * numAnchors + a];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        if (bestScore < confThresh) continue;

        // Box in letterboxed-input coordinates, center-format
        float cx = rawOutput[0 * numAnchors + a];
        float cy = rawOutput[1 * numAnchors + a];
        float w  = rawOutput[2 * numAnchors + a];
        float h  = rawOutput[3 * numAnchors + a];

        // Undo letterbox: subtract padding, divide by scale, to get original-image coords
        //float x1 = (cx - w / 2.f - lb.padX) / lb.scale;
        //float y1 = (cy - h / 2.f - lb.padY) / lb.scale;
        //float x2 = (cx + w / 2.f - lb.padX) / lb.scale;
        //float y2 = (cy + h / 2.f - lb.padY) / lb.scale;

        float x1 = (cx - w / 2.f) * 1920./640.;
        float y1 = (cy - h / 2.f) * 1080./640.;
        float x2 = (cx + w / 2.f) * 1920./640.;
        float y2 = (cy + h / 2.f) * 1080./640.;

        // Clamp to image bounds
        x1 = std::clamp(x1, 0.f, static_cast<float>(1920));
        y1 = std::clamp(y1, 0.f, static_cast<float>(1080));
        x2 = std::clamp(x2, 0.f, static_cast<float>(1920));
        y2 = std::clamp(y2, 0.f, static_cast<float>(1080));

        boxes.emplace_back(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    // Class-aware NMS: cv::dnn::NMSBoxes doesn't separate by class on its own,
    // so either loop per class, or use NMSBoxesBatched with classIds.
    std::vector<int> keepIndices;
    cv::dnn::NMSBoxesBatched(boxes, scores, classIds, confThresh, nmsThresh, keepIndices);

    std::vector<Detection> detections;
    detections.reserve(keepIndices.size());
    for (int idx : keepIndices) {
        Detection d;
        d.box = boxes[idx];
        d.confidence = scores[idx];
        d.class_id = classIds[idx];
        //d.class_name = classNames[classIds[idx]];
        d.class_name = "";
        detections.push_back(d);
    }

    return detections;
}