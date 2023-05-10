#include <rendering/renderer.h>
#include <rendering/shader.h>
#include <rendering/scene.h>
#include <rendering/camera.h>
#include <rendering/iglrenderable.h>

#include <glm/gtc/type_ptr.hpp>



void Renderer::renderScene(Scene* scene) {
    //setup viewMatrices ecc
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
    const std::vector<IglRenderable*> renderablesPtrs = scene->getObjects();
    for (uint i = 0; i < renderablesPtrs.size(); i++) {
        IglRenderable* renderablePtr = renderablesPtrs[i];
        if (renderablePtr == nullptr) continue;
        const glm::mat4 modelMatrix = scene->getModelMatrixOf(i);
        //set model matrix in shader
        renderablePtr->draw();
    }
}