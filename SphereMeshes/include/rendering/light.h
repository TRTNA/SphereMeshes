#pragma once

#include <glm/vec3.hpp>

struct Light {
    Light(glm::vec3 vec);
    Light();
    glm::vec3 vec;
};