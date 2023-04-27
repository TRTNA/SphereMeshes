#pragma once

#include <glm/vec3.hpp>
#include <utils/types.h>

struct Capsuloid {
    float factor;
    glm::vec3 S0toS1;
    float sqrdL;
    uint s0, s1;
    Capsuloid() = default;
    Capsuloid(uint s0, uint s1);
    Capsuloid(uint s0, uint s1, float factor);
    void setFactor(float factor);
};

std::ostream& operator<<(std::ostream& ost, const Capsuloid& val);
std::istream& operator>>(std::istream& ost, Capsuloid& val);