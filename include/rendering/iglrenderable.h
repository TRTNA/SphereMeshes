#pragma once

#include <rendering/shader.h>

class IglRenderable {
    public:
        virtual void Draw(const Shader& shader) = 0;
};