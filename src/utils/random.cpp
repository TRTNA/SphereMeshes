#include <utils/random.h>
#include <glm/gtx/string_cast.hpp>
#include <random>

glm::vec3 generatePosition(std::pair<float, float> xRange, std::pair<float, float> yRange, std::pair<float, float> zRange) {
    float x = xRange.first + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(xRange.second-xRange.first)));
    float y = yRange.first + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(yRange.second-yRange.first)));
    float z = zRange.first + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(zRange.second-zRange.first)));
    return glm::vec3(x, y, z);
}

glm::vec3 generatePosition(const AABB& aabb) {
    return generatePosition(aabb.getXRange(), aabb.getYRange(), aabb.getZRange());
}

float generateFloat(float lowerLim, float upperLim) {
        return lowerLim+ static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(upperLim-lowerLim)));
}