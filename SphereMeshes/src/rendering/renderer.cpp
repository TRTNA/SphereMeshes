#include <rendering/renderer.h>
#include <rendering/shader.h>
#include <rendering/scene.h>
#include <rendering/camera.h>
#include <rendering/iglrenderable.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_inverse.hpp>


#include <utils/plane.h>


Renderer::Renderer(Shader* shader) : shader(shader) {
    shader->Use();

    /////////FIXME: oscenità da sistemare ma per ora ho solo un tipo di shader
    materialTypeToSubroutineIdx.emplace(MaterialType::BLINN_PHONG_DOUBLE_SIDE, glGetSubroutineIndex(shader->Program, GL_FRAGMENT_SHADER, "doubleSidedShadedColoring"));
    materialTypeToSubroutineIdx.emplace(MaterialType::BLINN_PHONG, glGetSubroutineIndex(shader->Program, GL_FRAGMENT_SHADER, "shadedColoring"));
    materialTypeToSubroutineIdx.emplace(MaterialType::FLAT, glGetSubroutineIndex(shader->Program, GL_FRAGMENT_SHADER, "flatColoring"));
    materialTypeToSubroutineIdx.emplace(MaterialType::NORMAL, glGetSubroutineIndex(shader->Program, GL_FRAGMENT_SHADER, "normalColoring"));
    materialTypeToSubroutineIdx.emplace(MaterialType::DIMENSIONALITY, glGetSubroutineIndex(shader->Program, GL_FRAGMENT_SHADER, "diffuseColoring"));
    /////////

    glGetProgramStageiv(shader->Program, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &activeSubroutineCount);
}


void Renderer::renderScene(Scene* scene) {
    //GLOBAL PARAMS SETUP
    shader->Use();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glClearColor(backgroundColor.x, backgroundColor.y, backgroundColor.z, 1.0f);

    Camera const* camera = scene->getCamera();
    Light const* light = scene->getLight();
    const glm::mat4 projectionMatrix = camera->getProjectionMatrix();
    const glm::mat4 viewMatrix = camera->getViewMatrix();
    glUniformMatrix4fv(glGetUniformLocation(shader->Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(glGetUniformLocation(shader->Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
    GLfloat ambCol[3] = {ambientColor.x, ambientColor.y, ambientColor.z};
    glUniform3fv(glGetUniformLocation(shader->Program, "ambientColor"), 1, ambCol);
    glm::vec3 vLightDir = glm::vec3(viewMatrix * glm::vec4(light->vec, 0.0));
    glUniform3fv(glGetUniformLocation(shader->Program, "vLightDir"), 1, glm::value_ptr(vLightDir));

    glUniform1i(glGetUniformLocation(shader->Program, "backFaceCulling"), backfaceCulling);

    const std::vector<IglRenderable*> renderablesPtrs = scene->getObjects();
    for (uint i = 0; i < renderablesPtrs.size(); i++) {
        IglRenderable* renderablePtr = renderablesPtrs[i];
        if (renderablePtr == nullptr || ! scene->isObjectEnabled(i)) continue;

        //MATERIAL SETUP
        const Material* mat = scene->getMaterialOf(i);
        GLfloat diffCol[3] = {mat->diffuseColor.x, mat->diffuseColor.y, mat->diffuseColor.z};
        glUniform3fv(glGetUniformLocation(shader->Program, "diffuseColor"), 1, diffCol);
        GLfloat specCol[3] = {mat->specularColor.x, mat->specularColor.y, mat->specularColor.z};
        glUniform3fv(glGetUniformLocation(shader->Program, "specularColor"), 1, specCol);
        glUniform1f(glGetUniformLocation(shader->Program, "shininess"), mat->shininess);
        glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, activeSubroutineCount, &materialTypeToSubroutineIdx.at(mat->type));


        // TRANSFORM SETUP
        const glm::mat4 modelMatrix = *(scene->getModelMatrixOf(i));
        glUniformMatrix4fv(glGetUniformLocation(shader->Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
        glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(viewMatrix*modelMatrix));
        glUniformMatrix3fv(glGetUniformLocation(shader->Program, "normalMatrix"), 1, GL_FALSE, glm::value_ptr(normalMatrix));

        renderablePtr->draw();

        // SECOND PASS: SHADOWING
        if (shadowing && mat->shadowing) {
            glm::mat4 shadowProjMatrix = glm::mat4(1.0f);
            glm::vec3 N = shadowPlane.getNormal();
            glm::vec3 V = shadowPlane.getOrigin();
            glm::vec3 D = glm::vec3(0.0f, 1.0f, 0.0f);
            float NdotD = glm::dot(N, D);
            shadowProjMatrix[0][1] = -N.x / NdotD;
            shadowProjMatrix[1][1] = (1.0f - N.y) / NdotD;
            shadowProjMatrix[2][1] = -N.z / NdotD;
            shadowProjMatrix[3][1] = (glm::dot(V, N) / NdotD) + 0.1f;
            
            shadowProjMatrix = shadowProjMatrix * modelMatrix;
            glUniformMatrix4fv(glGetUniformLocation(shader->Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(shadowProjMatrix));
            glUniform3fv(glGetUniformLocation(shader->Program, "diffuseColor"), 1, shadowPlaneColor);
            glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, activeSubroutineCount, &materialTypeToSubroutineIdx.at(MaterialType::FLAT));
            renderablePtr->draw();

        }
    }
}

void Renderer::enableShadowing(Plane plane, glm::vec3 planeColor) {
    shadowing = true;
    shadowPlane = plane;
    for (int i = 0; i < 3; i++) {
        shadowPlaneColor[i] = planeColor[i] * 0.25f;
    }
}

void Renderer::disableShadowing() {
    shadowing = false;
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