#include <rendering/pointlight.h>

PointLight::PointLight(glm::vec3 pos) : pos(pos){}
PointLight::PointLight() : PointLight(glm::vec3(0.0f)) {}