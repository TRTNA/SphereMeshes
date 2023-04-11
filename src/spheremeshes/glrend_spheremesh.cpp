#include <spheremeshes/glrend_spheremesh.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <utils/random.h>
#include <utils/pointcloud.h>

#include <iostream>

using std::cout;
using std::endl;

using std::vector;


GlRendSphereMesh::GlRendSphereMesh(std::vector<Sphere> &pSpheres, std::vector<Edge> &pEdges, std::vector<Triangle> &pTriangles) : SphereMesh(pSpheres, pEdges, pTriangles)
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

void GlRendSphereMesh::Draw(const Shader &shader)
{   
    unsigned int pointNo = 10000;
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    for (const auto& e : edges) {
        const Sphere& s1 = spheres.at(e.first);
        const Sphere& s2 = spheres.at(e.second);
        float maxRadius = std::max(s1.radius, s2.radius);
        PointCloud PC;
        std::pair<float, float> xRange(std::min(s1.center.x, s2.center.x) - maxRadius, std::max(s1.center.x, s2.center.x) + maxRadius);
        std::pair<float, float> yRange(std::min(s1.center.y, s2.center.y) - maxRadius, std::max(s1.center.y, s2.center.y) + maxRadius);
        std::pair<float, float> zRange(std::min(s1.center.z, s2.center.z) - maxRadius, std::max(s1.center.z, s2.center.z) + maxRadius);
        for (size_t i = 0; i < pointNo; i++) {
            PC.addPoint(generatePoint(xRange, yRange, zRange));
        }

        std::vector<glm::vec3> points = PC.getPoints();
        glBufferData(GL_ARRAY_BUFFER, pointNo * sizeof(glm::vec3), points.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);

        
        glUniform3fv(glGetUniformLocation(shader.Program, "capsA"), 1, glm::value_ptr(s1.center));
        glUniform3fv(glGetUniformLocation(shader.Program, "capsB"), 1, glm::value_ptr(s2.center));
        glUniform1f(glGetUniformLocation(shader.Program, "radiusA"), s1.radius);
        glUniform1f(glGetUniformLocation(shader.Program, "radiusB"), s2.radius);

        glDrawArrays(GL_POINTS, 0, pointNo);

    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

GlRendSphereMesh::~GlRendSphereMesh()
{
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }
}
