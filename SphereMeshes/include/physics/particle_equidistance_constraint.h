#pragma once

#include <physics/constraint.h>
#include <physics/particle.h>

class ParticleEquidistanceConstraint : public Constraint
{
    private:
        Particle *p1, *p2;
        float dist;
public:
    ParticleEquidistanceConstraint(Particle *p1, Particle *p2, float dist);
    void enforce() override;
};
