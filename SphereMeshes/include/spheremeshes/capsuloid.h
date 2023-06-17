#pragma once

#include <glm/vec3.hpp>
#include <utils/types.h>

/* 
    Struct that models a capsuloid (a capsule with spheres of possibly different radius)
    Spheres at the vertices are stored as uint that are indexes of the spheres vector in the sphere mesh

    Factor, S0toS1 and sqrdL are constant features of the capsuloid that are chaced for performance:
        - factor is float that indicates how to modify the parameter k, which is used to find the projection of a point on the capsuloid segment
        - S0toS1 is the vector from sphere 0 to sphere 1
        - sqrdL is the square magnitude of S0toS1
    BEWARE: features must updated when data about the spheres are available
*/
struct Capsuloid {
    float factor;
    glm::vec3 S0toS1;
    float sqrdL;
    uint s0, s1;
    Capsuloid() = default;
    Capsuloid(uint s0, uint s1);
    Capsuloid(uint s0, uint s1, float factor);
};

std::ostream& operator<<(std::ostream& ost, const Capsuloid& val);
std::istream& operator>>(std::istream& ost, Capsuloid& val);