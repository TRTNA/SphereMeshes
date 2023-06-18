#pragma once

#include <physics/physicalobject.h>
#include <physics/particle.h>
#include <spheremeshes/spheremesh.h>
#include <glm/glm.hpp>
#include <vector>

class Constraint;

class PhysicsSphereMesh : public PhysicalObject
{
private:
    std::shared_ptr<SphereMesh> sphereMesh;
    std::vector<glm::vec3> localSpaceVectors;
    glm::vec3 localSpaceBarycenter;
    std::vector<Constraint*> constraints;
    float totalMass;

    glm::mat4 modelMatrix;
    void setup();
    glm::mat4 computeModelMatrix();

public:
    PhysicsSphereMesh(std::shared_ptr<SphereMesh> sphereMesh);
    std::vector<Particle> particles;
    std::vector<float> radii;
    void addConstraint(Constraint* constraint);
    glm::mat4 getModelMatrix() const;
    virtual void addForce(const glm::vec3 &forceVec) override;
    virtual void enforceConstraints() override;
    virtual float getMass() const override;
    virtual void timeStep() override;
};

