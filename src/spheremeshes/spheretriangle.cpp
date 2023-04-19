#include <spheremeshes/SphereTriangle.h>

#include <string>

using std::array;

SphereTriangle::SphereTriangle(uint v0, uint v1, uint v2) : vertices(array<uint, 3>{{v0, v1, v2}}) {}
SphereTriangle::SphereTriangle(uint v0, uint v1, uint v2, glm::mat3 projMat) : vertices(array<uint, 3>{{v0, v1, v2}}), projectorMatrix(projMat) {}

SphereTriangle::SphereTriangle(const std::array<uint, 3>& pVertices) : vertices(array<uint, 3>{pVertices}) {}
SphereTriangle::SphereTriangle(const std::array<uint, 3>& pVertices, glm::mat3 projMat) : vertices(array<uint, 3>{pVertices}), projectorMatrix(projMat) {}


std::ostream& operator<<(std::ostream& ost, const SphereTriangle& val) {
    ost << val.vertices.at(0) << " ";
    ost << val.vertices.at(1) << " ";
    ost << val.vertices.at(2);
    return ost;
}

std::istream& operator>>(std::istream& ist, SphereTriangle& val) {
    ist >> val.vertices.at(0) >> val.vertices.at(1) >> val.vertices.at(2);
    return ist;
}

void SphereTriangle::setProjectorMatrix(glm::mat3 mat) {
    projectorMatrix = mat;
}


