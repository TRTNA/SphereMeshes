#include <rendering/renderer.h>
#include <rendering/shader.h>
#include <rendering/scene.h>
#include <rendering/camera.h>
#include <rendering/iglrenderable.h>

Renderer::Renderer(Shader shader) : shader(shader) {}

void Renderer::renderScene(Scene* scene) {
    //setup viewMatrices ecc
    const std::vector<IglRenderable*> renderablesPtrs = scene->getObjects();
    for (uint i = 0; i < renderablesPtrs.size(); i++) {
        IglRenderable* renderablePtr = renderablesPtrs[i];
        if (renderablePtr == nullptr) continue;
        const glm::mat4 modelMatrix = scene->getModelMatrixOf(i);
        //set model matrix in shader
        renderablePtr->draw();
    }
}