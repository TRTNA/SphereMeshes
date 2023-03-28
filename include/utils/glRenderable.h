#ifndef _GLRENDERABLE_H
#define _GLRENDERABLE_H

#include <utils/shader.h>

class glRenderable {
    public:
        virtual void Draw(const Shader& shader) = 0;
};
#endif