#include <rendering/material.h>

using glm::vec3;

Material::Material() : Material(vec3(0.0f), vec3(0.0f), 0.0f, MaterialType::FLAT) {}
Material::Material(glm::vec3 diffuseColor, glm::vec3 specularColor, float shininess, MaterialType type, bool shadowing) : diffuseColor(diffuseColor), specularColor(specularColor), shininess(shininess), type(type), shadowing(shadowing) {}