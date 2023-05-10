#include <rendering/scene.h>
#include <cassert>

#include <rendering/iglrenderable.h>

uint Scene::addObject(IglRenderable* objPtr, glm::mat4 modelMatrix) {
    objects.push_back(objPtr);
    modelMatrices.push_back(modelMatrix);
    return objects.size() - 1;
}
void Scene::removeObject(uint idx) {
    assert(idx < objects.size());
    objects.at(idx) = nullptr;
    modelMatrices.at(idx) = glm::mat4(0.0f);
}

const std::vector<IglRenderable*> Scene::getObjects() const {
    return objects;
}

glm::mat4 Scene::getModelMatrixOf(uint idx) const {
    assert (idx < modelMatrices.size());
    return modelMatrices.at(idx);
}
