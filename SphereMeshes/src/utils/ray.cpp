#include <utils/ray.h>
#include <spheremeshes/sphere.h>
#include <spheremeshes/point.h>

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
    vec3 rayToSphere = sphere.center - ray.pos;

    //ray dir is perpendicular to or facing away from the vector that connects ray pos to sphere
    const float tP = dot(rayToSphere, ray.dir);
    if (tP < 0.0f) return false;

    const vec3 distVector = sphere.center - (ray.pos + tP*ray.dir);
    const float sqrdDist = dot(distVector, distVector);
    const float sphereRadiusSqrd = sphere.radius * sphere.radius;
    
    if (dot(distVector, distVector) > sphereRadiusSqrd) return false;
    
    const float tI = glm::sqrt(sphereRadiusSqrd - sqrdDist);
    outIntersPoint.pos = ray.pos + (tP - tI)*ray.dir;
    outIntersPoint.normal = glm::normalize(outIntersPoint.pos - sphere.center);
    return true;
}