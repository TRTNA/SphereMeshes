#include <cloth/particle.h>

#include <exception>

Particle::Particle(glm::vec3 pos, glm::vec3 normal, float massKg) : pos(pos), normal(normal), force(glm::vec3(0.0f)), lastPos(pos), massKg(massKg), pinned(false) {}

Particle::Particle() : Particle(glm::vec3(0.0f), glm::vec3(0.0f), 0.0f) {}

void Particle::timeStep()
{
    if (! pinned) {
        glm::vec3 temp = pos;
        glm::vec3 acceleration = force / massKg;
        pos = pos + (pos - lastPos) * (1.0f - DAMPING) + acceleration * TIME_STEP_SQRD;
        lastPos = temp;
    }

}
void Particle::addForce(const glm::vec3 &force)
{
    this->force += force;
}