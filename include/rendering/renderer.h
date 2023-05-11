#pragma once

#include <glm/glm.hpp>

class Scene;
class Shader;

class Renderer {
    private:
        bool backfaceCulling = false;
        glm::vec3 ambientColor = glm::vec3(0.0f);
        Shader shader;
    public:
        void renderScene(Scene* scene);
        void setBackfaceCulling(bool state);
        void setAmbientColor(glm::vec3 ambientColor);
        void setShader(Shader shader);
};