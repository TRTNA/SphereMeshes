#include <rendering/renderer.h>
#include <rendering/shader.h>
#include <rendering/scene.h>
#include <rendering/camera.h>
#include <rendering/iglrenderable.h>

#include <glm/gtc/type_ptr.hpp>

Renderer::Renderer(Shader* shader) : shader(shader) {}


void Renderer::renderScene(Scene* scene) {
    //GLOBAL PARAMS SETUP
    shader->Use();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glClearColor(backgroundColor.x, backgroundColor.y, backgroundColor.z, 1.0f);

    const glm::mat4 projectionMatrix = scene->camera->getProjectionMatrix();
    const glm::mat4 viewMatrix = scene->camera->getViewMatrix();
    glUniformMatrix4fv(glGetUniformLocation(shader->Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(glGetUniformLocation(shader->Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
    GLfloat ambCol[3] = {ambientColor.x, ambientColor.y, ambientColor.z};
    glUniform3fv(glGetUniformLocation(shader->Program, "ambientColor"), 1, ambCol);
    glUniform3fv(glGetUniformLocation(shader->Program, "vLightPos"), 1, glm::value_ptr(glm::vec3(viewMatrix * glm::vec4(scene->light->pos, 1.0))));

    glUniform1i(glGetUniformLocation(shader->Program, "backFaceCulling"), backfaceCulling);

    const std::vector<IglRenderable*> renderablesPtrs = scene->getObjects();
    for (uint i = 0; i < renderablesPtrs.size(); i++) {
        IglRenderable* renderablePtr = renderablesPtrs[i];
        if (renderablePtr == nullptr) continue;
        if (! scene->isEnabled(i)) continue;

        //MATERIAL SETUP
        const Material* mat = scene->getMaterialOf(i);
        GLfloat diffCol[3] = {mat->diffuseColor.x, mat->diffuseColor.y, mat->diffuseColor.z};
        glUniform4fv(glGetUniformLocation(shader->Program, "diffuseColor"), 1, diffCol);
        GLfloat specCol[3] = {mat->specularColor.x, mat->specularColor.y, mat->specularColor.z};
        glUniform4fv(glGetUniformLocation(shader->Program, "specularColor"), 1, specCol);
        glUniform1f(glGetUniformLocation(shader->Program, "shininess"), mat->shininess);

        // TRANSFORM SETUP
        const glm::mat4 modelMatrix = *(scene->getModelMatrixOf(i));
        glUniformMatrix4fv(glGetUniformLocation(shader->Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
        glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(viewMatrix) * glm::mat3(modelMatrix)));
        glUniformMatrix3fv(glGetUniformLocation(shader->Program, "normalMatrix"), 1, GL_FALSE, glm::value_ptr(normalMatrix));

        renderablePtr->draw();
    }
}


void Renderer::setBackfaceCulling(bool state) {
    backfaceCulling = state;
}

void Renderer::setAmbientColor(glm::vec3 ambientColor) {
    this->ambientColor = ambientColor;
}

void Renderer::setBackgroundColor(glm::vec3 backgroundColor) {
    this->backgroundColor = backgroundColor;
}


void Renderer::setShader(Shader* shader) {
    this->shader = shader;
}