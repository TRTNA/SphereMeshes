#pragma once

#include <rendering/shader.h>

class IglRenderable {
    public:
        virtual void draw() = 0;
};