#pragma once

#include <cloth/cloth.h>
#include <vector>

typedef unsigned int uint;

class RenderableCloth : public Cloth
{
public:
    RenderableCloth(uint dim, float dist);
    ~RenderableCloth();
    void enforceConstraints();
    void draw();
        void updateBuffers();
    void updateNormals();
private:
    uint VAO, EBO, VBO;
    std::vector<uint> indices;



};