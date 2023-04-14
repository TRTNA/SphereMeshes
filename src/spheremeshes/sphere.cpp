#include <spheremeshes/sphere.h>
#include <glm/gtx/string_cast.hpp>

Sphere::Sphere(glm::vec3 pCenter, float pRadius) : center(pCenter), radius(pRadius) {}

Sphere::Sphere() : center(glm::vec3(0.0f)), radius(0.0f) {}

std::ostream& operator<<(std::ostream& ost, const Sphere& val) {
    ost << "Sphere(";
    ost << glm::to_string(val.center);
    ost << "; ";
    ost << val.radius << ")";
    return ost;
}

 
Sphere computeBoundingSphere(const Sphere& s1, const Sphere& s2)
{
    glm::vec3 c2c1 = s2.center - s1.center;
    float c2c1mag = glm::length(c2c1);
    float R = (s2.radius + s1.radius + c2c1mag) / 2.0f;
    glm::vec3 C = s1.center + c2c1*(R - s1.radius) / c2c1mag;
    return Sphere(C, R);
}