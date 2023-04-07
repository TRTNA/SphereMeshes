#include <utils/pointcloud.h>

void PointCloud::addPoint(glm::vec3 point) {
    points_.push_back(point);
}
void PointCloud::addPoints(std::vector<glm::vec3> points) {
    points_.insert(points_.end(), points.begin(), points.end());
}

std::vector<glm::vec3> PointCloud::getPoints() const {
    return points_;
}