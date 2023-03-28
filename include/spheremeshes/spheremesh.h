#ifndef _SPHEREMESH_H
#define _SPHEREMESH_H

#include <vector>
#include <string>
#include <ostream>


#include <spheremeshes/sphere.h>
#include <spheremeshes/edge.h>
#include <spheremeshes/triangle.h>

#include <utils/model.h>
#include <utils/shader.h>


class SphereMesh {
    public:
    std::vector<Sphere> spheres;
    std::vector<Edge> edges;
    std::vector<Triangle> triangles;
    SphereMesh() = default;
    SphereMesh(std::vector<Sphere>& pSpheres, std::vector<Edge>& pEdges, std::vector<Triangle>& pTriangles);
    ~SphereMesh() = default;
    void addSphere(const Sphere& phere);
    void addEdge(const Edge& edge);
    void addTriangle(const Triangle& triangle);
    std::string toString() const;
};

std::ostream& operator<<(std::ostream& ost, const SphereMesh& sm);

void renderGL(const SphereMesh& sm, const Model& sphereModel, const Shader& shader);

#endif