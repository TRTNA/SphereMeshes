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

std::ostream& operator<<(std::ostream& ost, const SphereMesh& sm) {
    ost << sm.toString();
    return ost;
}


