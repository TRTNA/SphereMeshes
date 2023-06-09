#pragma once

#include <glm/glm.hpp>

enum class MaterialType {
    FLAT,
    BLINN_PHONG,
    DIMENSIONALITY,
    NORMAL
};

struct Material {
    Material();
    Material(glm::vec3 diffuseColor, glm::vec3 specularColor, float shininess, MaterialType type);
    glm::vec3 diffuseColor;
    glm::vec3 specularColor;
    float shininess;
    MaterialType type;
};