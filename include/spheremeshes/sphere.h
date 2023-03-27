#ifndef _SPHERE_H
#define _SPHERE_H

#include <glm/vec3.hpp>
#include <ostream>

struct Sphere {
    Sphere();
    Sphere(glm::vec3 pCenter, float pRadius);
    glm::vec3 center;
    float radius;
};

std::ostream& operator<<(std::ostream& ost, const Sphere& val);

#endif