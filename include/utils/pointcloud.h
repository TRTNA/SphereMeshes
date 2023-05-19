#pragma once

#include <vector>

#include <glm/vec3.hpp>

#include <spheremeshes/point.h>
#include <rendering/shader.h>


class SphereMesh; // <-- forward declaration

class PointCloud {
    private:
        std::vector<DimensionalityPoint> points;
        void addPoint(const SphereMesh& sphereMesh);
    public:
        void repopulate(const unsigned int nPoints, const SphereMesh& sphereMesh);
        void setPoints(std::vector<DimensionalityPoint> points);
        void clear();
        unsigned int getPointsNumber() const;
        const void* pointerToData() const;
};
