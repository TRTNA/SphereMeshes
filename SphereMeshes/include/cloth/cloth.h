#pragma once

#include <vector>
#include <utility>
#include <string>

#include <glm/glm.hpp>
#include <physics/physicalobject.h>
#include <physics/particle_equidistance_constraint.h>

typedef unsigned int uint;
class Particle;

class Cloth : public PhysicalObject {
    protected:
        Particle* particles;
        std::vector<Constraint*> constraints;
        uint dim;
        uint dimSqrd;
        float dist;
    public:
        Cloth(uint dim, float dist, const glm::vec3& translation = glm::vec3(0.0f));
        ~Cloth();
        void transform(const glm::mat4& matrix);
        std::string toString() const;
        float getMass() const override;
        uint getParticles(Particle*& outParticles);
        void addForce(const glm::vec3& force) override;
        void timeStep() override;
        void enforceConstraints() override;
        void addConstraint(Constraint* constraint);

};