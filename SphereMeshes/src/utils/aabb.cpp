#include <utils/aabb.h>    

AABB::AABB(floatRange xRange, floatRange yRange, floatRange zRange) : xRange_(xRange), yRange_(yRange), zRange_(zRange) {}
AABB::AABB(float xMin, float xMax, float yMin, float yMax, float zMin, float zMax) :
    xRange_(floatRange(xMin, xMax)), yRange_(floatRange(yMin, yMax)), zRange_(floatRange(zMin, zMax)) {}

floatRange AABB::getXRange() const {
    return xRange_;
}
floatRange AABB::getYRange() const {
    return yRange_;
}
floatRange AABB::getZRange() const {
    return zRange_;
}