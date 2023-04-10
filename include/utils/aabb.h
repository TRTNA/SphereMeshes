#ifndef _AABB_H
#define _AABB_H

#include <utility>

typedef std::pair<float, float> floatRange;
class AABB {
    private:
    floatRange xRange_;
    floatRange yRange_;
    floatRange zRange_;
    public:
    AABB(floatRange xRange, floatRange yRange, floatRange zRange);
    AABB(float xMin, float xMax, float yMin, float yMax, float zMin, float zMax);
    floatRange getXRange() const;
    floatRange getYRange() const;
    floatRange getZRange() const;

};

#endif