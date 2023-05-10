#include <rendering/renderablecloth.h>
#include <glad/glad.h>
#include <spheremeshes/point.h>
#include <utils/common.h>
#include <cloth/particle.h>

#include <vector>
#include <iostream>

using std::vector;

RenderableCloth::RenderableCloth(uint dim, float dist) : Cloth(dim, dist)
{

    glGenVertexArrays(1, &this->VAO);
    glGenBuffers(1, &this->VBO);
    glGenBuffers(1, &this->EBO);
    // Indices do not change, so EBO is initialized here and never updated
    glBindVertexArray(this->VAO);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
    triangulateSquareGrid(dim, indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint), indices.data(), GL_STATIC_DRAW);
    updateNormals();

    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
    glBufferData(GL_ARRAY_BUFFER, dim * dim * sizeof(Particle), this->particles, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (GLvoid *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (GLvoid *)offsetof(Particle, normal));

}

RenderableCloth::~RenderableCloth()
{
    if (VAO)
    {
        glDeleteVertexArrays(1, &this->VAO);
        glDeleteBuffers(1, &this->VBO);
        glDeleteBuffers(1, &this->EBO);
    }
}
void RenderableCloth::enforceConstraints()
{
    Cloth::enforceConstraints();
    updateNormals();
    updateBuffers();
}
void RenderableCloth::draw()
{
    glBindVertexArray(this->VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void RenderableCloth::updateBuffers()
{
    glBindVertexArray(this->VAO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, dim * dim * sizeof(Particle), particles);
    glBindVertexArray(0);
}

void RenderableCloth::updateNormals()
{
    // zeroing all normals
    for (size_t i = 0; i < dim * dim; i++)
    {
        particles[i].normal = glm::vec3(0.0f);
    }

    // iterate over triangles
    for (size_t tri = 0; tri <= indices.size() - 3; tri += 3)
    {
        Particle &p1 = particles[indices.at(tri)];
        Particle &p2 = particles[indices.at(tri + 1)];
        Particle &p3 = particles[indices.at(tri + 2)];

        glm::vec3 normal = glm::cross(p2.pos - p1.pos, p3.pos - p1.pos);
        p1.normal += normal;
        p2.normal += normal;
        p3.normal += normal;
    }

    // normalize all normals
    for (size_t i = 0; i < dim * dim; i++)
    {
        particles[i].normal = glm::normalize(particles[i].normal);
    }
}


void RenderableCloth::timeStep() {
    Cloth::timeStep();
    updateNormals();
    updateBuffers();
}



