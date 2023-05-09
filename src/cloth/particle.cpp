#include <cloth/particle.h>

Particle::Particle(glm::vec3 pos, glm::vec3 normal, float massKg) : pos(pos), normal(normal), lastPos(pos), massKg(massKg) {}

