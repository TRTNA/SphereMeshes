#pragma once

#include <physics/constraint.h>

class Particle;
class Cloth;
class SphereMesh;


#include <vector>


class ParticleSphereMeshCollisionConstraint : public Constraint {
    private:
        SphereMesh *sphereMesh;
        Particle *particle;
    public:
        ParticleSphereMeshCollisionConstraint(SphereMesh* sphereMesh, Particle *particle);
        void enforce() override;
};

class ClothSphereMeshCollisionConstraint : public Constraint {
    private:
        SphereMesh *sphereMesh;
        Cloth *cloth;
        std::vector<ParticleSphereMeshCollisionConstraint> constraints;
    public:
        ClothSphereMeshCollisionConstraint(SphereMesh* sphereMesh, Cloth *cloth);
        void enforce() override;
};