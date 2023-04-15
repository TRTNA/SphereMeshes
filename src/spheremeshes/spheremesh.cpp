#include <spheremeshes/spheremesh.h>

#include <iostream>
#include <sstream>
#include <array>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "spheremesh.h"

using std::clog;
using std::endl;

using std::vector;
using std::array;
using std::ostream;
using std::stringstream;

SphereMesh::SphereMesh(vector<Sphere>& pSpheres, vector<Edge>& pEdges, vector<Triangle>& pTriangles)
     : spheres(std::move(pSpheres)), edges(std::move(pEdges)), triangles(std::move(pTriangles)) 
{
    clog << "Created a sphere mesh:\n";
    clog << "- Spheres:\t" << spheres.size() << "\n";
    clog << "- Edges:\t" << edges.size() <<"\n";
    clog << "- Triangles:\t" << triangles.size() <<"\n";
}

void SphereMesh::addSphere(const Sphere& sphere) {
    spheres.emplace_back(sphere.center, sphere.radius);
}

void SphereMesh::addEdge(const Edge& edge) {
    edges.emplace_back(edge.first, edge.second);
}

void SphereMesh::addTriangle(const Triangle& triangle) {
    triangles.emplace_back(triangle.vertices);
}

void SphereMesh::updateBoundingSphere()
{
    boundingSphere = computeBoundingSphere(spheres);
}
std::string SphereMesh::toString() const
{
    stringstream ss;
    ss << "Spheres:\n";
    for(size_t i = 0; i < spheres.size(); i++) {
        ss << i << " " << spheres.at(i) << "\n";
    }
    ss << "\n";
    ss << "Edges:\n";
    for(size_t i = 0; i < edges.size(); i++) {
        ss << i << " " << edges.at(i) << "\n";
    }
    ss << "\n";
    ss << "Triangles:\n";
    for(size_t i = 0; i < triangles.size(); i++) {
        ss << i << " " << triangles.at(i) << "\n";
    }
    return ss.str();
}
Point SphereMesh::pushOutsideOneCapsule(uint capsuleIndex, const glm::vec3 &pos, int &dimensionality)
{
    const Edge& edge = edges.at(capsuleIndex);
    const Sphere& A = spheres.at(edge.first);
    const Sphere& B = spheres.at(edge.first);

    const glm::vec3 BminusA = B.center - A.center;
    const float BminusAsqrd = glm::dot(BminusA, BminusA);
    float k = glm::dot(pos - A.center, BminusA) / BminusAsqrd;
    const float factor = (A.radius - B.radius) / length(BminusA);
    k -= factor * length(pos - (A.center + k*BminusA));
    const float clampedK = glm::clamp(k, 0.0f, 1.0f);

    const glm::vec3 C = A.center + clampedK*BminusA;
    const glm::vec3 CtoPos = pos - C;
    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);
    const float interpRadius = A.radius * (1.0f - clampedK) + B.radius * clampedK;

    //pos is outside the capsule, dimensionality is -1 (not pushed out)
    if (CtoPossqrd > interpRadius*interpRadius) {
        dimensionality = -1;
        return Point(pos, glm::vec3(0.0f));
    }

    //if we are here, pos is inside the capsule
    //dimensionality depends on K value
    //if clampedK == k then pos is inside the cylinder, so dimensionality = 1
    //else pos is inside one of the spheres, so dimensionality = 0
    dimensionality = k == clampedK ? 1 : 0;
    const glm::vec3 normal = glm::normalize(CtoPos);
    return Point(glm::vec3(C + interpRadius*normal), normal);
}

std::ostream& operator<<(std::ostream& ost, const SphereMesh& sm) {
    ost << sm.toString();
    return ost;
}


