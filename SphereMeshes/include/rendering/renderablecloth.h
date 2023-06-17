#pragma once

#include <cloth/cloth.h>
#include <vector>
#include <rendering/iglrenderable.h>
#include <memory>

typedef unsigned int uint;

struct Point;

class RenderableCloth : public IglRenderable
{
public:
    RenderableCloth(std::shared_ptr<Cloth> clothPtr);
    ~RenderableCloth();
    void draw() override;
    void enforceConstraints();
    void updateBuffers();
    void updateNormals();
private:
    uint VAO, EBO, VBO;
    std::vector<uint> indices;
    std::shared_ptr<Cloth> clothPtr;
};