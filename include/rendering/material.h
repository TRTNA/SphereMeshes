#pragma once

#include <glm/glm.hpp>

struct Material {
    glm::vec3 diffuseColor;
    glm::vec3 specularColor;
    float shininess;
};