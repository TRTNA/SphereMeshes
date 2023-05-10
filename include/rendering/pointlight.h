#pragma once

#include <glm/vec3.hpp>

struct PointLight {
    PointLight(glm::vec3 pos);
    PointLight();
    glm::vec3 pos;
};