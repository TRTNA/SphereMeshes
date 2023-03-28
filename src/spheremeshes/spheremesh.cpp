#include <spheremeshes/spheremesh.h>

#include <iostream>
#include <sstream>
#include <array>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

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

std::string SphereMesh::toString() const {
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

void renderGL(const SphereMesh& sm, const Model& sphereModel, const Shader& shader) {
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    vector<glm::vec3> vertices;
    //Drawing spheres
    for(const auto& s : sm.spheres) {
        // Setting transformation matrix so that sphere will be rendered in s.center and uniformly scaled with factor s.radius
        modelMatrix = glm::mat4(1.0f);
        modelMatrix = glm::translate(modelMatrix, s.center);
        modelMatrix = glm::scale(modelMatrix, glm::vec3(s.radius));
        glUniformMatrix4fv(glGetUniformLocation(shader.Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));

        // Draw the sphere
        sphereModel.Draw();

        vertices.push_back(s.center);

    }
    
    //Drawing triangles faces
    vector<unsigned int> indices;
    for(const auto& t : sm.triangles) {
        const array<U32, 3>& triVertices = t.vertices;
        indices.push_back(triVertices.at(0));
        indices.push_back(triVertices.at(1));
        indices.push_back(triVertices.at(2));
    }

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    glBindVertexArray(VAO);
    modelMatrix = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);


    //Drawing edges
    vector<unsigned int> edgesIndices;
    for(const auto& e : sm.edges) {
        edgesIndices.push_back(e.first);
        edgesIndices.push_back(e.second);
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgesIndices.size() * sizeof(unsigned int), edgesIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    glBindVertexArray(VAO);
    modelMatrix = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glDrawElements(GL_LINES, edgesIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);



}

