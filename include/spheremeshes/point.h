#pragma once

struct Point {
    glm::vec3 pos;
    glm::vec3 normal;
};

struct ColoredPoint : public Point {
    glm::vec3 color;
};

