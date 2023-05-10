#pragma once

class Scene;
class Shader;

class Renderer {

    public:
        Shader shader;
        glm::mat4 projectionMatrix;
        glm::mat4 viewMatrix;
        void renderScene(Scene* scene);
};