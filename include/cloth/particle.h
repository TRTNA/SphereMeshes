#pragma once

#include <glm/vec3.hpp>

struct Particle {
    glm::vec3 pos;
    glm::vec3 lastPos;
    glm::vec3 normal;
    float massKg;
    Particle(glm::vec3 pos, glm::vec3 normal, float massKg);
    Particle() = default;
};