#ifndef _GLRENDERABLE_H
#define _GLRENDERABLE_H

#include <rendering/shader.h>

class IglRenderable {
    public:
        virtual void Draw(const Shader& shader) = 0;
};
#endif