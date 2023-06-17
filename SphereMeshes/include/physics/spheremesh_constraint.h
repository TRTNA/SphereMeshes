#pragma once

#include <physics/constraint.h>
#include <physics/plane_constraint.h>
#include <vector>


class Particle;
class Cloth;
class SphereMesh;
class PhysicsSphereMesh;
class Plane;



class ParticleSphereMeshCollisionConstraint : public Constraint {
    private:
        SphereMesh *sphereMesh;
        Particle *particle;
        float attritionDamping;
        float verticalOffset;
    public:
        ParticleSphereMeshCollisionConstraint(SphereMesh* sphereMesh, Particle *particle, float attritionDamping = 0.3f, float verticalOffset = 0.25f);
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

class SphereMeshPlaneConstraint : public Constraint {
    private:
        PhysicsSphereMesh* physSphereMesh;
        Plane* plane;
        std::vector<ParticlePlaneConstraint> constraints;
    public:
        SphereMeshPlaneConstraint(PhysicsSphereMesh* physSphereMesh, Plane* plane);
        void enforce() override;
};