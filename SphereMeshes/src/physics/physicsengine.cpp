#include <physics/physicsengine.h>
#include <cloth/cloth.h>
#include <spheremeshes/point.h>

#include <utils/time.h>

#include <algorithm>

CollisionData::CollisionData(glm::vec3 point, glm::vec3 normal) : point(point), normal(normal), collided(true){}

void PhysicsEngine::start()
{
    paused = false;
    virtualTime = getTimeSeconds();
}
void PhysicsEngine::pause()
{
    paused = true;
}

double PhysicsEngine::getVirtualTime() const
{
    return virtualTime;
}

void PhysicsEngine::synchronizeWithWallTime(double wallTime)
{
    virtualTime = wallTime;
}

bool PhysicsEngine::isPaused() const
{
    return paused;
}

void PhysicsEngine::timeStep()
{
    if (paused)
        return;

    //physical step
    for (auto &obj : objects)
    {
        obj->timeStep();
    }
    // constraint enforcement
    const uint maxTries = 5U;
    for (uint tries = 0; tries < maxTries; tries++)
    {
        for (auto &obj : objects)
        {
            obj->enforceConstraints();
        }
    }
    virtualTime += PHYSICS_TIME_STEP;
}
void PhysicsEngine::addObject(PhysicalObject *objectPtr)
{
    objects.push_back(objectPtr);
}
void PhysicsEngine::removeObject(PhysicalObject *objectPtr)
{
    auto newPastEnd = std::remove(objects.begin(), objects.end(), objectPtr);
    if (newPastEnd != objects.end())
        objects.erase(newPastEnd, objects.end());
}

CollisionData PhysicsEngine::checkCollision(const Particle *particle, const SphereMesh *sm) const
{
    int dimensionality = -1;
    Point point = sm->pushOutside(particle->getPos(), dimensionality);
    if (dimensionality != -1)
    {
        return CollisionData(point.pos, point.normal);
    }
    else
    {
        return CollisionData();
    }
}
