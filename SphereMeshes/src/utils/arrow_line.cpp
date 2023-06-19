#include <utils/arrow_line.h>
#include <glad/glad.h>

using std::vector;

ArrowLine::~ArrowLine() noexcept
{
    glDeleteVertexArrays(1, &this->VAO);
    glDeleteBuffers(1, &this->VBO);
}

ArrowLine::ArrowLine(ArrowLine &&move) noexcept : points(std::move(move.points)), n(move.n)
{
}

ArrowLine &ArrowLine::operator=(ArrowLine &&move) noexcept
{
    if (move.VAO)
    {
        points = std::move(move.points);
    }
    else
    {
        VAO = 0;
    }
    return *this;
}

ArrowLine::ArrowLine(glm::vec3 direction) : n(100)
{
    glGenVertexArrays(1, &this->VAO);
    glGenBuffers(1, &this->VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);


    points.emplace_back((-n * 0.5f) * direction);
    points.emplace_back((n * 0.5f) * direction);


    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(glm::vec3), &this->points[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(glm::vec3),
        (GLvoid *)0);
    glBindVertexArray(0);
}

void ArrowLine::draw()
{
    glBindVertexArray(this->VAO);
    glDrawArrays(GL_LINES, 0, 2);
    glBindVertexArray(0);
}