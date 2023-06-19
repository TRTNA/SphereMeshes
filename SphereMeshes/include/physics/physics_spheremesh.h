#pragma once

#include <physics/physicalobject.h>
#include <physics/particle.h>
#include <spheremeshes/spheremesh.h>
#include <glm/glm.hpp>
#include <vector>

class Constraint;

static enum class PhysicsSphereMeshType {
      GENERIC,
      ONE_SPHERE,
      TWO_SPHERES
};

class PhysicsSphereMesh : public PhysicalObject
{
private:
    std::shared_ptr<SphereMesh> sphereMesh;
    std::vector<glm::vec3> localSpaceVectors;
    std::vector<Constraint*> constraints;
    float totalMass;
    PhysicsSphereMeshType type;

    float twoSpheresDist = 0.0f;

    glm::mat4 modelMatrix;
    void setup();
    glm::mat4 computeModelMatrix();
    glm::vec3 computeWorldSpaceBarycenter() const;
    void nSpheresEnforce();
    void twoSphereEnforce();

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

