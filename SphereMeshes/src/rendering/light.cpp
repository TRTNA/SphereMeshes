#include <rendering/light.h>

Light::Light(glm::vec3 vec) : vec(vec){}
Light::Light() : Light(glm::vec3(0.0f)) {}