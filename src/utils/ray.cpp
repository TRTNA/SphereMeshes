#include <utils/ray.h>

#include <glm/gtx/string_cast.hpp>

using glm::vec3;
using std::string;

Ray::Ray(vec3 dir) : Ray::Ray(vec3(0.0f), dir) {}
Ray::Ray(vec3 pos, vec3 dir) : dir(glm::normalize(dir)), pos(pos) {}
string Ray::toString() const {
    return "Ray(pos: " + glm::to_string(pos) + "; " + glm::to_string(dir) + ")";
}