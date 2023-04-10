#ifndef _RANDOM_H
#define _RANDOM_H

#include <utils/aabb.h>

#include <random>
#include <limits>
#include <glm/vec3.hpp>

glm::vec3 generatePoint(std::pair<float, float> xRange, std::pair<float, float> yRange, std::pair<float, float> zRange) {
    float x = xRange.first + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(xRange.second-xRange.first)));
    float y = yRange.first + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(yRange.second-yRange.first)));
    float z = zRange.first + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(zRange.second-zRange.first)));
    return glm::vec3(x, y, z);


}




#endif