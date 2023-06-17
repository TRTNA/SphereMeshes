#pragma once

#include <glm/vec3.hpp>



class PhysicalObject {
    public:
    virtual void addForce(const glm::vec3& forceVec) = 0;
    virtual void enforceConstraints() = 0;
    virtual float getMass() const = 0;
    virtual void timeStep() = 0;
    virtual ~PhysicalObject() {};
};