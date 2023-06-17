#include <physics/spheremesh_constraint.h>
#include <physics/plane_constraint.h>
#include <physics/physics_spheremesh.h>

#include <spheremeshes/spheremesh.h>
#include <physics/physicsconstants.h>
#include <cloth/cloth.h>
#include <physics/particle.h>

using std::vector;

typedef unsigned int uint;

ParticleSphereMeshCollisionConstraint::ParticleSphereMeshCollisionConstraint(SphereMesh *sphereMesh, Particle *particle, float attritionDamping, float verticalOffset) : sphereMesh(sphereMesh), particle(particle), attritionDamping(attritionDamping), verticalOffset(verticalOffset) {}

void ParticleSphereMeshCollisionConstraint::enforce()
{
    int dimensionality = -1;
    Point point = sphereMesh->pushOutside(particle->getPos(), dimensionality);
    if (dimensionality != -1)
    {
        point.pos += point.normal * verticalOffset;
        glm::vec3 v = (point.pos - particle->lastPos) / PHYSICS_TIME_STEP;
        glm::vec3 vPerp = point.normal * glm::dot(point.normal, v);
        glm::vec3 vPar = v - vPerp;
        // velocity damping
        v = vPerp + vPar * attritionDamping;
        particle->setPos(particle->lastPos + v * PHYSICS_TIME_STEP);
    }
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

SphereMeshPlaneConstraint::SphereMeshPlaneConstraint(PhysicsSphereMesh *physSphereMesh, Plane *plane) : physSphereMesh(physSphereMesh), plane(plane) {
    for (auto& p : physSphereMesh->particles) {
        constraints.emplace_back(plane, &p);
    }
}

void SphereMeshPlaneConstraint::enforce()
{
    for(auto& c : constraints) {
        c.enforce();
    }
}