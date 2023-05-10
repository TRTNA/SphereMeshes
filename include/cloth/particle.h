#pragma once

#include <glm/vec3.hpp>

const float DAMPING = 0.01f;
const float TIME_STEP = 1.0f/30.0f; //30fps
const float TIME_STEP_SQRD = TIME_STEP * TIME_STEP;

struct Particle {
    glm::vec3 pos;
    glm::vec3 lastPos;
    glm::vec3 normal;
    glm::vec3 force;
    float massKg;
    bool pinned;
    Particle(glm::vec3 pos, glm::vec3 normal, float massKg);
    Particle();
    void timeStep();
    void addForce(const glm::vec3& force);
};