#include <spheremeshes/glrend_spheremesh.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

using std::cout;
using std::endl;

using std::vector;

Model* GlRendSphereMesh::sphereModel = nullptr;

GlRendSphereMesh::GlRendSphereMesh(std::vector<Sphere> &pSpheres, std::vector<Edge> &pEdges, std::vector<Triangle> &pTriangles) : SphereMesh(pSpheres, pEdges, pTriangles)
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    updateVAO();
}

void GlRendSphereMesh::Draw(const Shader &shader)
{
    glm::mat4 modelMatrix = glm::mat4(1.0f);

    //draw spheres
     for(const auto& s : this->spheres) {
        // Setting transformation matrix so that sphere will be rendered in s.center and uniformly scaled with factor s.radius
        modelMatrix = glm::mat4(1.0f);
        modelMatrix = glm::translate(modelMatrix, s.center);
        modelMatrix = glm::scale(modelMatrix, glm::vec3(s.radius));
        glUniformMatrix4fv(glGetUniformLocation(shader.Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));

        // Draw the sphere
        sphereModel->Draw();
    }

    //resetting matrix
    glBindVertexArray(VAO);
    modelMatrix = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));

    //draw triangles
    glDrawElements(GL_TRIANGLES, triangleIndicesNo, GL_UNSIGNED_INT, 0);

    //draw edges
    void* edgesIndicesStartPtr = (void*) (triangleIndicesNo * sizeof(unsigned int));
    glDrawElements(GL_LINES, edgesIndicesNo, GL_UNSIGNED_INT, edgesIndicesStartPtr);
    
    glBindVertexArray(0);
}

void GlRendSphereMesh::updateVAO()
{
    vector<glm::vec3> vertices;
    for(const auto& s : this->spheres) {
        vertices.push_back(s.center);

    }
    vector<unsigned int> indices;
    for(const auto& t : this->triangles) {
        indices.push_back(t.vertices.at(0));
        indices.push_back(t.vertices.at(1));
        indices.push_back(t.vertices.at(2));
    }
    triangleIndicesNo = indices.size();

    for(const auto& e : this->edges) {
        indices.push_back(e.first);
        indices.push_back(e.second);
    }
    edgesIndicesNo = indices.size() - triangleIndicesNo;

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0);
}

GlRendSphereMesh::~GlRendSphereMesh()
{
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
    }
}
