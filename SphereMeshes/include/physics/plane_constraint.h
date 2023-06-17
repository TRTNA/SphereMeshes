#pragma once

#include <physics/constraint.h>
#include <physics/particle.h>
#include <cloth/cloth.h>
#include <utils/plane.h>

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
