#pragma once

#include <physics/physicalobject.h>
#include <physics/physicsconstants.h>
#include <spheremeshes/spheremesh.h>
#include <physics/particle.h>
#include <glm/glm.hpp>
#include <vector>

struct CollisionData {
    CollisionData() = default;
    CollisionData(glm::vec3 point, glm::vec3 normal);
    bool collided = false;
    glm::vec3 normal;
    glm::vec3 point;
};

class Plane;

class PhysicsEngine
{
private:
    double virtualTime = 0.0f;
    std::vector<PhysicalObject *> objects;
    bool paused = false;

public:
    void synchronizeWithWallTime(double wallTime);
    double getVirtualTime() const;
    void start();
    void pause();
    bool isPaused() const;
    void timeStep();
    void addObject(PhysicalObject *objectPtr);
    void removeObject(PhysicalObject *objectPtr);
    CollisionData handleCollision(const Particle* particle, const SphereMesh* sphereMesh) const;
    CollisionData handleCollision(const SphereMesh* sphereMesh, const Plane* plane) const;
};