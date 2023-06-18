#include <utils/plane.h>
#include <glm/geometric.hpp>
#include <glm/gtc/epsilon.hpp>

#include <utils/types.h>

using glm::vec3;

Plane::Plane(glm::vec3 origin, glm::vec3 normal) : origin(origin), normal(normal)
{
}
glm::vec3 Plane::getNormal() const
{
    return normal;
}

glm::vec3 Plane::getOrigin() const
{
    return origin;
}
void Plane::setNormal(glm::vec3 normal)
{
    this->normal = normal;
}

void Plane::setOrigin(glm::vec3 origin)
{
    this->origin = origin;
}

bool Plane::contains(glm::vec3 point) const
{
    vec3 originToPoint = point - origin;
    float dot = glm::dot(normal, originToPoint);
    return dot < 0.0001f || dot > -0.0001f;
}

bool Plane::isPerpendicular(glm::vec3 vec) const
{
    vec3 cross = glm::cross(vec, normal);
    return glm::all(glm::epsilonEqual(cross, glm::vec3(0.0f), EPSILON));
}
