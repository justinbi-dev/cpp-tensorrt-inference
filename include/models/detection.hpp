#pragma once

struct BoundingBox {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct Detection {
    BoundingBox box;
    float confidence;
    int class_id;
};
