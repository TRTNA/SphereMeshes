#pragma once

#include <glm/glm.hpp>
#include <string>
struct Sphere;
struct Point;

static const float EPSILON = 0.0001f;

struct Ray {
    Ray(glm::vec3 dir);
    Ray(glm::vec3 pos, glm::vec3 dir);
    std::string toString() const;
    glm::vec3 pos;
    //normalized dir
    glm::vec3 dir;
};

bool intersects(const Ray& ray, const Sphere& sphere, Point& outIntersPoint);