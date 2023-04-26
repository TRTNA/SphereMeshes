#ifndef _SphereTriangle_H
#define _SphereTriangle_H

#include <utils/types.h>
#include <glm/mat3x3.hpp>
#include <array>
#include <ostream>

//TODO: triangle should store two matrices, one for the bottom plane and one for the upper one
//TODO: nei calcoli sui triangoli servono: S0S1, S0S2, normale, correzione

struct SphereTriangle {
    std::array<U32, 3> vertices;
    glm::mat3 projectorMatrix;
    glm::vec3 S0S1, S0S2;
    glm::vec3 N;
    glm::vec3 invN;
    SphereTriangle() = default;
    SphereTriangle(uint v0, uint v1, uint v2);
    SphereTriangle(uint v0, uint v1, uint v2, glm::mat3 projMat);
    SphereTriangle(const std::array<uint, 3>& pVertices);
    SphereTriangle(const std::array<uint, 3>& pVertices, glm::mat3 projMat);
    void setProjectorMatrix(glm::mat3 mat);
};

std::ostream& operator<<(std::ostream& ost, const SphereTriangle& val);
std::istream& operator>>(std::istream& ist, SphereTriangle& val);

#endif