#pragma once

#include <glm/glm.hpp>

enum class MaterialType {
    FLAT,
    BLINN_PHONG,
    BLINN_PHONG_DOUBLE_SIDE,
    DIMENSIONALITY,
    NORMAL
};

struct Material {
    Material();
    Material(glm::vec3 diffuseColor, glm::vec3 specularColor, float shininess, MaterialType type, bool shadowing = false);
    glm::vec3 diffuseColor;
    glm::vec3 specularColor;
    float shininess;
    bool shadowing;
    MaterialType type;
};