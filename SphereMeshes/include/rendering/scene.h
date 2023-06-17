#pragma once

#include <vector>
#include <glm/glm.hpp>

#include <rendering/light.h>
#include <rendering/camera.h>
#include <rendering/material.h>


typedef unsigned int uint;
class Shader;
class IglRenderable;

class Scene {
    private:
        std::vector<IglRenderable*> objects;
        std::vector<glm::mat4*> modelMatrices;
        std::vector<Material*> materials;
        std::vector<bool> enabled;
        Camera* camera;
        Light* light;
    public:
        Scene() = default;
        Scene(Camera* camera, Light* light);

        uint addObject(IglRenderable* objPtr, glm::mat4* modelMatrix, Material* mat);
        void removeObject(uint idx);
        const std::vector<IglRenderable*> getObjects();
        void disableObject(uint idx);
        void enableObject(uint idx);
        bool isObjectEnabled(uint idx) const;

        glm::mat4* getModelMatrixOf(uint idx);
        Material* getMaterialOf(uint idx);

        Camera const* getCamera() const;
        Light const* getLight() const;

};