#pragma once

#include <utils/types.h>
#include <glm/mat3x3.hpp>
#include <array>
#include <ostream>

/*
    Struct that models a triangular prismoid, with three spheres with possibily different radius in its vertices
    Spheres at the vertices are stored as uint that are indexes of the spheres vector in the sphere mesh

    S0S1, S0S2, upperProjMatrix, lowerProjMatrix and planeN are constant features of the sphere triangle that are cached for performance:
    - projection matrices (upper and lower) are a matrices to find the direction of projection of a point (expressed as a vector from a point on the plane to that point)
        wrt the plane. These matrices take into account the fact that the plane passing through the centers of the spheres and the plane that is tangent
        to the spheres can be not parallel. 
    - upperProjMatrix is a projection matrix used with points that are in the positive semispace defined by the plane that pass through the centers of the spheres
    - lowerProjMatrix is a projection matrix used with points that are in the negative semispace.
    - planeN is the normal (in world space) of the plane that passes through the vertices (centers of the spheres) of the triangles.
    - S0S1 is the vector that goes from the center of sphere 0 to the one of sphere 1
    - S0S2 is the vector that goes from the center of sphere 0 to the one of sphere 2
    BEWARE: features must updated when data about the spheres are available
*/
struct SphereTriangle {
    std::array<U32, 3> vertices;
    glm::vec3 S0S1, S0S2;
    glm::mat3 upperProjMatrix;
    glm::mat3 lowerProjMatrix;
    glm::vec3 planeN;
    SphereTriangle() = default;
    SphereTriangle(uint v0, uint v1, uint v2);
    SphereTriangle(const std::array<uint, 3>& pVertices);
};

std::ostream& operator<<(std::ostream& ost, const SphereTriangle& val);
std::istream& operator>>(std::istream& ist, SphereTriangle& val);
