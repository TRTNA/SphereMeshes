#include <physics/particle.h>
#include <physics/physicsconstants.h>

#include <exception>

Particle::Particle(glm::vec3 pos, glm::vec3 normal, float massKg, bool pinned) : PhysicalObject(), pos(pos), normal(normal), force(glm::vec3(0.0f)), lastPos(pos), massKg(massKg), pinned(pinned) {}

Particle::Particle() : Particle(glm::vec3(0.0f), glm::vec3(0.0f), 0.0f, false) {}

void Particle::timeStep()
{
    if (!pinned)
    {
        glm::vec3 temp = pos;
        glm::vec3 acceleration = force / massKg;
        acceleration += DEFAULT_GRAVITY_ACC;
        pos = pos + (pos - lastPos) * (1.0f - DAMPING) + acceleration * PHYSICS_TIME_STEP_SQRD;
        lastPos = temp;
    }
}
void Particle::addForce(const glm::vec3 &force)
{
    this->force += force;
}
void Particle::resetForce()
{

    force = glm::vec3(0.0f);
}

void Particle::setNormal(glm::vec3 normal)
{
    this->normal = normal;
}

void Particle::enforceConstraints() {
    
}


glm::vec3 Particle::getNormal() const
{
    return normal;
}

void Particle::displace(glm::vec3 vec) {
    if (! pinned) pos += vec;
}

void Particle::setPos(glm::vec3 pos)
{
    this->pos = pos;
}

void Particle::setLastPos(glm::vec3 lastPos)
{
    this->lastPos = lastPos;
}

glm::vec3 Particle::getPos() const
{
    return pos;
}

glm::vec3 Particle::getLastPos() const
{
    return lastPos;
}

void Particle::setMass(float massKg)
{
    this->massKg = massKg;
}

float Particle::getMass() const
{
    return massKg;
}

void Particle::pin()
{
    pinned = true;
}

void Particle::unpin()
{
    pinned = false;
}

bool Particle::isPinned() const
{
    return pinned;
}
