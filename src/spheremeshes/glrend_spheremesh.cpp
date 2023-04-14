#include <spheremeshes/glrend_spheremesh.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <utils/random.h>
#include <utils/pointcloud.h>

#include <iostream>
#include <spheremeshes/glrend_spheremesh.h>

using std::cout;
using std::endl;
using std::shared_ptr;
using std::vector;


GlRendSphereMesh::GlRendSphereMesh(std::vector<Sphere> &pSpheres, std::vector<Edge> &pEdges, std::vector<Triangle> &pTriangles, unsigned int pPointsNumber) : SphereMesh(pSpheres, pEdges, pTriangles), pointsNumber(pPointsNumber)
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    for (const auto& e : edges) {
        const Sphere& s1 = spheres.at(e.first);
        const Sphere& s2 = spheres.at(e.second);
        float maxRadius = std::max(s1.radius, s2.radius);
        shared_ptr<PointCloud> PC = std::make_shared<PointCloud>();
        std::pair<float, float> xRange(std::min(s1.center.x, s2.center.x) - maxRadius, std::max(s1.center.x, s2.center.x) + maxRadius);
        std::pair<float, float> yRange(std::min(s1.center.y, s2.center.y) - maxRadius, std::max(s1.center.y, s2.center.y) + maxRadius);
        std::pair<float, float> zRange(std::min(s1.center.z, s2.center.z) - maxRadius, std::max(s1.center.z, s2.center.z) + maxRadius);
        for (size_t i = 0; i < pointsNumber; i++) {
            PC->addPoint(generatePoint(xRange, yRange, zRange));
        }
        pcs.push_back(PC);
    }
}

GlRendSphereMesh::GlRendSphereMesh(unsigned int pPointsNumber) : SphereMesh(), pointsNumber(pPointsNumber) {}

//TODO il punto deve essere spinto fuori in CPU non in vertex shader
//
void GlRendSphereMesh::Draw(const Shader &shader)
{   
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    for (size_t i = 0; i < edges.size(); i++) {
        const Sphere& s1 = spheres.at(edges.at(i).first);
        const Sphere& s2 = spheres.at(edges.at(i).second);
        float maxRadius = std::max(s1.radius, s2.radius);
        shared_ptr<PointCloud>& PC = pcs.at(i);
        
        std::vector<glm::vec3> points = PC->getPoints();
        glBufferData(GL_ARRAY_BUFFER, pointsNumber * sizeof(glm::vec3), points.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);

        
        glUniform3fv(glGetUniformLocation(shader.Program, "capsA"), 1, glm::value_ptr(s1.center));
        glUniform3fv(glGetUniformLocation(shader.Program, "capsB"), 1, glm::value_ptr(s2.center));
        glUniform1f(glGetUniformLocation(shader.Program, "radiusA"), s1.radius);
        glUniform1f(glGetUniformLocation(shader.Program, "radiusB"), s2.radius);

        glDrawArrays(GL_POINTS, 0, pointsNumber);

    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GlRendSphereMesh::setPointsNumber(unsigned int pPointsNumber)
{
    this->pointsNumber = pPointsNumber;
}

void GlRendSphereMesh::regeneratePoints()
{
    pcs.clear();
    for (const auto& e : edges) {
        const Sphere& s1 = spheres.at(e.first);
        const Sphere& s2 = spheres.at(e.second);
        float maxRadius = std::max(s1.radius, s2.radius);
        shared_ptr<PointCloud> PC = std::make_shared<PointCloud>();
        std::pair<float, float> xRange(std::min(s1.center.x, s2.center.x) - maxRadius, std::max(s1.center.x, s2.center.x) + maxRadius);
        std::pair<float, float> yRange(std::min(s1.center.y, s2.center.y) - maxRadius, std::max(s1.center.y, s2.center.y) + maxRadius);
        std::pair<float, float> zRange(std::min(s1.center.z, s2.center.z) - maxRadius, std::max(s1.center.z, s2.center.z) + maxRadius);
        for (size_t i = 0; i < pointsNumber; i++) {
            PC->addPoint(generatePoint(xRange, yRange, zRange));
        }
        pcs.push_back(PC);
    }

}
GlRendSphereMesh::~GlRendSphereMesh()
{
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }
}
