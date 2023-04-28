#pragma once

#include <glm/glm.hpp>
#include <string>

struct Ray {
    Ray(glm::vec3 dir);
    Ray(glm::vec3 pos, glm::vec3 dir);
    std::string toString() const;
    glm::vec3 pos;
    //normalized dir
    glm::vec3 dir;
};