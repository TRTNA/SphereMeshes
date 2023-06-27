#include <physics/plane_constraint.h>
#include <glm/glm.hpp>

#include <utils/plane.h>
#include <physics/particle.h>
#include <cloth/cloth.h>
#include <spheremeshes/spheremesh.h>
#include <physics/physics_spheremesh.h>
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


SpherePlaneConstraint::SpherePlaneConstraint(const Plane* plane, Particle* particle, float radius) : plane(plane), particle(particle), radius(radius) {}

void SpherePlaneConstraint::enforce() {
    vec3 planeToCenter = particle->getPos() - plane->getOrigin();
    float dist = glm::dot(planeToCenter, plane->getNormal());
    if (dist < radius) {
        //sphere is below the plane
        //needs displacement
        float displ = radius - dist;
        vec3 displacementVector = displ * plane->getNormal();
        particle->displace(displacementVector);
    }
}

PhysSphereMeshPlaneConstraint::PhysSphereMeshPlaneConstraint(const Plane* plane, PhysicsSphereMesh* physSphereMesh) {
    for (int i = 0; i < physSphereMesh->particles.size(); i++) {
        constraints.emplace_back(plane, &physSphereMesh->particles.at(i), physSphereMesh->radii.at(i));
    }
}
void PhysSphereMeshPlaneConstraint::enforce() {
    for (auto& c : constraints) {
        c.enforce();
    }
}