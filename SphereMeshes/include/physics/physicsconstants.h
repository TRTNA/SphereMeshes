#pragma once

#include <glm/vec3.hpp>

const float DAMPING = 0.01f;

const float PHYSICS_TIME_STEP = 1.0f / 30.0f; //30fps
const float PHYSICS_TIME_STEP_SQRD = PHYSICS_TIME_STEP * PHYSICS_TIME_STEP;

const glm::vec3 DEFAULT_GRAVITY_ACC = glm::vec3(0.0f, -9.8f, 0.0f);