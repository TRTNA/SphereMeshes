#ifndef _SphereTriangle_H
#define _SphereTriangle_H

#include <utils/types.h>
#include <glm/mat3x3.hpp>
#include <array>
#include <ostream>

struct SphereTriangle {
    std::array<U32, 3> vertices;
    glm::mat3 projectorMatrix;
    SphereTriangle() = default;
    SphereTriangle(uint v0, uint v1, uint v2);
    SphereTriangle(uint v0, uint v1, uint v2, glm::mat3 projMat);
    SphereTriangle(const std::array<uint, 3>& pVertices);
    SphereTriangle(const std::array<uint, 3>& pVertices, glm::mat3 projMat);
    //TODO triangolo si memorizza due matrici una per il piano sotto e una per quello sopra (la normale col vento)
    void setProjectorMatrix(glm::mat3 mat);
};

std::ostream& operator<<(std::ostream& ost, const SphereTriangle& val);
std::istream& operator>>(std::istream& ist, SphereTriangle& val);

#endif