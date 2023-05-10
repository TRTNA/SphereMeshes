#pragma once

#include <vector>
#include <glm/glm.hpp>

typedef unsigned int uint;
class Camera;
class Shader;
class IglRenderable;

class Scene {
    private:
        std::vector<IglRenderable*> objects;
        std::vector<glm::mat4> modelMatrices;
    public:
        Camera camera;
        uint addObject(IglRenderable* objPtr, glm::mat4 modelMatrix);
        void removeObject(uint idx);
        const std::vector<IglRenderable*> getObjects() const;
        glm::mat4 getModelMatrixOf(uint idx) const;
};