#include <rendering/renderableplane.h>

#include <utils/plane.h>
#include <utils/common.h>
#include <spheremeshes/point.h>

#include <glm/glm.hpp>

RenderablePlane::RenderablePlane(Plane plane, float dim)
{
    glGenVertexArrays(1, &this->VAO);
    glGenBuffers(1, &this->VBO);
    glGenBuffers(1, &this->EBO);

    // Indices do not change, so EBO is initialized here and never updated
    glBindVertexArray(this->VAO);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
    uint indices[6] = { 0, 1, 2, 1, 3, 2};

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(uint), indices, GL_STATIC_DRAW);

    glm::vec3 u = glm::normalize(glm::cross(plane.getNormal(), glm::vec3(1.0f, 0.0f, 0.0f)));
    glm::vec3 v = glm::normalize(glm::cross(plane.getNormal(), u));

    glm::vec3 vertices[8] = {
        plane.getOrigin() + glm::normalize(u + v) * (float)glm::sqrt(2.0) * (dim * 0.5f),
        plane.getNormal(),
        plane.getOrigin() + glm::normalize(u - v) * (float)glm::sqrt(2.0) * (dim * 0.5f),
        plane.getNormal(),
        plane.getOrigin() + glm::normalize(-u + v) * (float)glm::sqrt(2.0) * (dim * 0.5f),
        plane.getNormal(),
        plane.getOrigin() + glm::normalize(-u - v) * (float)glm::sqrt(2.0) * (dim*0.5f),
        plane.getNormal()
    };

    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(glm::vec3), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(glm::vec3), (void*)sizeof(glm::vec3));
}

RenderablePlane::~RenderablePlane()
{
    if (VAO)
    {
        glDeleteVertexArrays(1, &this->VAO);
        glDeleteBuffers(1, &this->VBO);
        glDeleteBuffers(1, &this->EBO);
    }
}

void RenderablePlane::draw()
{
    glBindVertexArray(this->VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
