#ifndef _POINTCLOUD_H
#define _POINTCLOUD_H

#include <vector>

#include <glm/vec3.hpp>

class PointCloud {
    private:
        std::vector<glm::vec3> points_;
    public:
        void addPoint(glm::vec3 point);
        void addPoints(std::vector<glm::vec3> points);
        std::vector<glm::vec3> getPoints() const;
};

#endif
