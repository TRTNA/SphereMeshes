#include <physics/particle_equidistance_constraint.h>

#include <glm/glm.hpp>

using glm::vec3;

ParticleEquidistanceConstraint::ParticleEquidistanceConstraint(Particle *p1, Particle *p2, float dist) : p1(p1), p2(p2), dist(dist)
{
}

void ParticleEquidistanceConstraint::enforce()
{
    vec3 v = p2->getPos() - p1->getPos();
    float currDist = glm::length(v);
    v /= currDist;
    float delta = currDist - dist;
    vec3 displacementVector = 0.5f * delta * v;
    p1->displace(displacementVector);
    p2->displace(-displacementVector);
}