#pragma once

#include <vector>
#include <glm/glm.hpp>

#include <rendering/pointlight.h>
#include <rendering/camera.h>
#include <rendering/material.h>


typedef unsigned int uint;
class Shader;
class IglRenderable;

class Scene {
    private:
        std::vector<IglRenderable*> objects;
        std::vector<glm::mat4> modelMatrices;
        std::vector<Material*> materials;
    public:
        Camera* camera;
        PointLight* light;
        uint addObject(IglRenderable* objPtr, glm::mat4 modelMatrix, Material* mat);
        void removeObject(uint idx);
        const std::vector<IglRenderable*> getObjects() const;
        glm::mat4 getModelMatrixOf(uint idx) const;
        Material* getMaterialOf(uint idx) const;
};