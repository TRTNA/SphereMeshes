#include <utils/ray.h>
#include <spheremeshes/sphere.h>

#include <glm/gtx/string_cast.hpp>

using glm::vec3;
using std::string;
using glm::dot;

Ray::Ray(vec3 dir) : Ray::Ray(vec3(0.0f), dir) {}
Ray::Ray(vec3 pos, vec3 dir) : dir(glm::normalize(dir)), pos(pos) {}
string Ray::toString() const {
    return "Ray(pos: " + glm::to_string(pos) + "; " + glm::to_string(dir) + ")";
}

bool intersects(const Ray& ray, const Sphere& sphere, Point& outIntersPoint) {
    //ray starting pos is inside the sphere, it intersects
    //FIXME inefficiente perchè in isInside vengono calcolate informazioni che sono utili anche qui
    if (isInside(ray.pos, sphere)) return true;

    //ray dir is perpendicular to or facing away from the vector that connects ray pos to sphere
    if (dot(sphere.center - ray.pos, ray.dir) <= 0.0f) return false;
}