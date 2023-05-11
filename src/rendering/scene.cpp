#include <rendering/scene.h>
#include <cassert>

#include <rendering/iglrenderable.h>

uint Scene::addObject(IglRenderable *objPtr, glm::mat4 *modelMatrix, Material *mat)
{
    objects.push_back(objPtr);
    modelMatrices.push_back(modelMatrix);
    materials.push_back(mat);
    enabled.push_back(true);
    return objects.size() - 1;
}
void Scene::removeObject(uint idx)
{
    assert(idx < objects.size());
    objects.at(idx) = nullptr;
    modelMatrices.at(idx) = nullptr;
    materials.at(idx) = nullptr;
    enabled.at(idx) = false;
}

const std::vector<IglRenderable *> Scene::getObjects()
{
    return objects;
}

glm::mat4 *Scene::getModelMatrixOf(uint idx)
{
    assert(idx < modelMatrices.size());
    return modelMatrices.at(idx);
}

Material *Scene::getMaterialOf(uint idx)
{
    assert(idx < modelMatrices.size());
    return materials.at(idx);
}

void Scene::disableObject(uint idx)
{
    assert(idx < modelMatrices.size());
    enabled.at(idx) = false;
}
void Scene::enableObject(uint idx)
{
    assert(idx < modelMatrices.size());
    enabled.at(idx) = true;
}

bool Scene::isEnabled(uint idx) const
{
    assert(idx < modelMatrices.size());
    return enabled.at(idx);
}
