#pragma once

#include <cloth/cloth.h>
#include <vector>
#include <rendering/iglrenderable.h>

typedef unsigned int uint;

struct Point;

class RenderableCloth : public Cloth, public IglRenderable
{
public:
    RenderableCloth(uint dim, float dist);
    ~RenderableCloth();
    void enforceConstraints();
    void draw() override;
    void timeStep();
    void updateBuffers();
    void updateNormals();
private:
    uint VAO, EBO, VBO;
    std::vector<uint> indices;



};