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
