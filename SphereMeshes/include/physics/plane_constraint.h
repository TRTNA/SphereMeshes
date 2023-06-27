#pragma once

#include <physics/constraint.h>

#include <vector>

class Particle;
class Plane;
class Cloth;
class Sphere;
class PhysicsSphereMesh;

typedef unsigned int uint;

class ParticlePlaneConstraint : public Constraint
{
private:
    Plane *plane;
    Particle *particle;

public:
    ParticlePlaneConstraint(Plane *plane, Particle *particle);
    void enforce() override;
};

class ClothPlaneConstraint : public Constraint
{
private:
    Plane *plane;
    Cloth *cloth;
    std::vector<ParticlePlaneConstraint> constraints;

public:
    ClothPlaneConstraint(Plane *plane, Cloth *cloth);
    void enforce() override;
};

class SpherePlaneConstraint : public Constraint {
private:
    const Plane* plane;
    Particle* particle;
    float radius;
public:
    SpherePlaneConstraint(const Plane* plane, Particle* particle, float radius);
    void enforce() override;
};

class PhysSphereMeshPlaneConstraint : public Constraint {
private:
    std::vector<SpherePlaneConstraint> constraints;
public:
    PhysSphereMeshPlaneConstraint(const Plane* plane, PhysicsSphereMesh* physSphereMesh);
    void enforce() override;
};
