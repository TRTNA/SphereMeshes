#include <rendering/renderablecloth.h>
#include <glad/glad.h>
#include <spheremeshes/point.h>
#include <utils/common.h>

#include <vector>
#include <iostream>

using std::vector;

RenderableCloth::RenderableCloth(uint dim, float dist) : Cloth(dim, dist)
{
    glGenVertexArrays(1, &this->VAO);
    glGenBuffers(1, &this->VBO);
    glGenBuffers(1, &this->EBO);


    updateBuffers();
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
    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
    glBufferData(GL_ARRAY_BUFFER, dim * dim * sizeof(Point), this->points, GL_STATIC_DRAW);
// Indices do not change, so EBO is initialized here and never updated
    indices.clear();
    triangulateSquareGrid(dim, indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint), indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Point), (GLvoid *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Point), (GLvoid *)offsetof(Point, normal));

    glBindVertexArray(0);
}
