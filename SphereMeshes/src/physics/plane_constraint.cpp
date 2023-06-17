#include <physics/plane_constraint.h>
#include <glm/glm.hpp>

using glm::vec3;

ParticlePlaneConstraint::ParticlePlaneConstraint(Plane *plane, Particle *particle) : plane(plane), particle(particle)
{
}

void ParticlePlaneConstraint::enforce()
{
    vec3 planeToParticle = particle->getPos() - plane->getOrigin();
    float dist = glm::dot(planeToParticle, plane->getNormal());
    if (dist < 0.0f) {
        //particle is below the plane
        //needs displacement
        dist *= -1.0f;
        vec3 displacementVector = plane->getNormal() * (dist + 0.01f*dist) ;
        particle->displace(displacementVector);
    }
}

ClothPlaneConstraint::ClothPlaneConstraint(Plane* plane, Cloth* cloth) : plane(plane), cloth(cloth) {
    Particle* particles;
    uint dim = cloth->getParticles(particles);
    for (size_t p = 0; p < dim*dim; p++) {
        constraints.emplace_back(plane, &particles[p]);
    }
}
void ClothPlaneConstraint::enforce() {
    for (auto& c : constraints) {
        c.enforce();
    }
}   