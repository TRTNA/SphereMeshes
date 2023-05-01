#include <rendering/renderablepointcloud.h>

void RenderablePointCloud::Draw(const Shader &shader)
{
    glBindVertexArray(this->VAO);
    glDrawArrays(GL_POINTS, 0, pointsNumber);
    glBindVertexArray(0);
}

RenderablePointCloud::~RenderablePointCloud()
{
    if (VAO)
    {

        glDeleteVertexArrays(1, &this->VAO);
        glDeleteBuffers(1, &this->VBO);
    }
}

RenderablePointCloud::RenderablePointCloud(std::shared_ptr<PointCloud> ptr) : pointCloud(ptr), pointsNumber(ptr->getPointsNumber())
{
    glGenVertexArrays(1, &this->VAO);
    glGenBuffers(1, &this->VBO);
    updateBuffers();
}

void RenderablePointCloud::updateBuffers()
{
    glBindVertexArray(this->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
    pointsNumber = pointCloud->getPointsNumber();
    glBufferData(GL_ARRAY_BUFFER, pointsNumber * sizeof(DimensionalityPoint), pointCloud->pointerToData(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DimensionalityPoint), (GLvoid *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(DimensionalityPoint), (GLvoid *)offsetof(DimensionalityPoint, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribIPointer(2, 1, GL_INT, sizeof(DimensionalityPoint), (GLvoid *)offsetof(DimensionalityPoint, dimensionality));

    glBindVertexArray(0);
}
