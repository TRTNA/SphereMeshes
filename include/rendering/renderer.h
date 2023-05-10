#pragma once

class Scene;
class Shader;

class Renderer {
    private:
        Shader shader;
    public:
        Renderer(Shader shader);
        void renderScene(Scene* scene);
};