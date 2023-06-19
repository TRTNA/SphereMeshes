#pragma once

#include <glm/glm.hpp>
#include <rendering/shader.h>
#include <utils/plane.h>

#include <unordered_map>

class Scene;
class Plane;
enum class MaterialType;

typedef unsigned int uint;

class Renderer {
    private:
        bool backfaceCulling = false;
        bool shadowing = false;
        Plane shadowPlane = Plane(glm::vec3(0.0f), glm::vec3(0.0f));
        float shadowPlaneColor[3] = { 0.f, 0.f, 0.f };
        glm::vec3 ambientColor = glm::vec3(0.0f);
        glm::vec3 backgroundColor = glm::vec3(0.0f);
        Shader* shader;
        std::unordered_map<MaterialType, uint> materialTypeToSubroutineIdx;
        int activeSubroutineCount;
    public:
        Renderer() = default;
        Renderer(Shader* shader);
        void renderScene(Scene* scene);
        void setBackfaceCulling(bool state);
        void setAmbientColor(glm::vec3 ambientColor);
        void setBackgroundColor(glm::vec3 backgroundColor);
        void setShader(Shader* shader);
        void enableShadowing(Plane plane, glm::vec3 planeColor);
        void disableShadowing();

};