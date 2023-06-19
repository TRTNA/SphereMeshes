#pragma once

#include <vector>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <spheremeshes/point.h>
#include <rendering/iglrenderable.h>
typedef unsigned int uint;

class ArrowLine : public IglRenderable {
    private:
        std::vector<glm::vec3> points;
        uint VAO;
        uint VBO;
        const int n;
    public:
        ArrowLine(const ArrowLine& copy) = delete;
        ArrowLine& operator=(const ArrowLine &) = delete;
        ~ArrowLine() noexcept;
        ArrowLine(ArrowLine&& move) noexcept;
        ArrowLine& operator=(ArrowLine&& move) noexcept;
        ArrowLine(glm::vec3 direction);
        void draw();

};