#pragma once

#include <cloth/cloth.h>
#include <vector>

typedef unsigned int uint;

struct Point;

class RenderableCloth : public Cloth
{
public:
    RenderableCloth(uint dim, float dist);
    ~RenderableCloth();
    void enforceConstraints();
    void draw();
        void timeStep();

private:
    uint VAO, EBO, VBO;
    std::vector<uint> indices;
    void updateBuffers();
    void updateNormals();


};