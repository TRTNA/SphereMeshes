#pragma once

#include <glm/vec3.hpp>
#include <physics/physicalobject.h>
#include <rendering/iglrenderable.h>

struct Particle : public PhysicalObject
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 lastPos;
    glm::vec3 force;
    float massKg;

    bool pinned;

    Particle(glm::vec3 pos, glm::vec3 normal, float massKg, bool pinned = false);
    Particle();
    
    void addForce(const glm::vec3 &force) override;
    void resetForce();

    void setNormal(glm::vec3 normal);
    glm::vec3 getNormal() const;

    void displace(glm::vec3 vec);
    void setPos(glm::vec3 pos);
    void setLastPos(glm::vec3 lastPos);
    glm::vec3 getPos() const;
    glm::vec3 getLastPos() const;

    void setMass(float massKg);
    float getMass() const override;

    void pin();
    void unpin();
    bool isPinned() const;

    void timeStep() override;
    void enforceConstraints() override;

};