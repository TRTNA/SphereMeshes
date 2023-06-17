#include <rendering/renderablecloth.h>
#include <glad/glad.h>
#include <spheremeshes/point.h>
#include <utils/common.h>
#include <physics/particle.h>

#include <vector>
#include <iostream>

using std::vector;
using std::shared_ptr;

RenderableCloth::RenderableCloth(shared_ptr<Cloth> clothPtr) : IglRenderable(), clothPtr(clothPtr)
{

    glGenVertexArrays(1, &this->VAO);
    glGenBuffers(1, &this->VBO);
    glGenBuffers(1, &this->EBO);

    // Indices do not change, so EBO is initialized here and never updated
    glBindVertexArray(this->VAO);
    Particle* particles = nullptr;
    uint dim = clothPtr->getParticles(particles);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
    triangulateSquareGrid(dim, indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint), indices.data(), GL_STATIC_DRAW);
    updateNormals();
    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
    glBufferData(GL_ARRAY_BUFFER, dim * dim * sizeof(particles[0]), particles, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(particles[0]), (GLvoid *)offsetof(Particle, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(particles[0]), (GLvoid *)offsetof(Particle, normal));
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

void RenderableCloth::draw()
{
    glBindVertexArray(this->VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void RenderableCloth::updateBuffers()
{
    Particle* particles = nullptr;
    uint dim = clothPtr->getParticles(particles);
    glBindVertexArray(this->VAO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, dim * dim * sizeof(particles[0]), particles);
    glBindVertexArray(0);
}

void RenderableCloth::updateNormals()
{
    Particle* particles = nullptr;
    uint dim = clothPtr->getParticles(particles);
    // zeroing all normals
    for (size_t i = 0; i < dim * dim; i++)
    {
        particles[i].normal = glm::vec3(0.0f);
    }

    for (size_t i = 0; i < dim - 1; i++)
    {
        for (size_t j = 0; j < dim - 1; j++)
        {
            Particle &p1 = particles[linearizedIndexSquareGrid(dim, i, j)];
            Particle &p2 = particles[linearizedIndexSquareGrid(dim, i, j + 1)];
            Particle &p3 = particles[linearizedIndexSquareGrid(dim, i + 1, j)];
            Particle &p4 = particles[linearizedIndexSquareGrid(dim, i + 1, j + 1)];
            
            glm::vec3 normal = glm::cross(p4.pos - p1.pos, p2.pos - p3.pos);
            p1.normal += normal;
            p2.normal += normal;
            p3.normal += normal;
            p4.normal += normal;
        }
    }
    
    // normalize all normals
    for (size_t i = 0; i < dim * dim; i++)
    {
        particles[i].normal = glm::normalize(particles[i].normal);
    }
}
