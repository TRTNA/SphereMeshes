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
    glBufferData(GL_ARRAY_BUFFER, pointsNumber * sizeof(ColoredPoint), pointCloud->pointerToData(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ColoredPoint), (GLvoid *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ColoredPoint), (GLvoid *)offsetof(ColoredPoint, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(ColoredPoint), (GLvoid *)offsetof(ColoredPoint, color));

    glBindVertexArray(0);
}
