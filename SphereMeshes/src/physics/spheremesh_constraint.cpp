#include <physics/spheremesh_constraint.h>

#include <spheremeshes/spheremesh.h>
#include <cloth/cloth.h>
#include <physics/particle.h>

using std::vector;

typedef unsigned int uint;

ParticleSphereMeshCollisionConstraint::ParticleSphereMeshCollisionConstraint(SphereMesh *sphereMesh, Particle *particle) : sphereMesh(sphereMesh), particle(particle) {}

void ParticleSphereMeshCollisionConstraint::enforce() {
    int dimensionality = -1;
    Point point = sphereMesh->pushOutside(particle->getPos(), dimensionality);
    particle->setPos(point.pos + point.normal*0.1f);
}

ClothSphereMeshCollisionConstraint::ClothSphereMeshCollisionConstraint(SphereMesh *sphereMesh, Cloth *cloth) : sphereMesh(sphereMesh), cloth(cloth)
{
    Particle *particles;
    uint dim = cloth->getParticles(particles);
    for (size_t i = 0; i < dim * dim; i++)
    {
        constraints.emplace_back(sphereMesh, &particles[i]);
    }
}
void ClothSphereMeshCollisionConstraint::enforce()
{
    for (auto &constraint : constraints)
    {
        constraint.enforce();
    }
}