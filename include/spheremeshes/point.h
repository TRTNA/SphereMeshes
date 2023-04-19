#pragma once

#include <glm/vec3.hpp>

struct Point {
    Point();
    Point(glm::vec3 pos, glm::vec3 normal);
    glm::vec3 pos;
    glm::vec3 normal;
};

struct DimensionalityPoint {
    DimensionalityPoint();
    DimensionalityPoint(glm::vec3 pos, glm::vec3 normal, int dimensionality);
    glm::vec3 pos;
    glm::vec3 normal;
    int dimensionality;
};

struct ColoredPoint : public Point {
    ColoredPoint();
    ColoredPoint(glm::vec3 pos, glm::vec3 normal, glm::vec3 color);
    ColoredPoint(const Point& point, glm::vec3 color);
    glm::vec3 color;
};

