#pragma once

#include <utils/aabb.h>
#include <glm/vec3.hpp>

glm::vec3 generatePosition(std::pair<float, float> xRange, std::pair<float, float> yRange, std::pair<float, float> zRange);

glm::vec3 generatePosition(const AABB& aabb);

float generateFloat(float lowerLim, float upperLim);




