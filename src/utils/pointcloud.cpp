#include <utils/pointcloud.h>
#include <spheremeshes/sphere.h>
#include <spheremeshes/spheremesh.h>

#include <glm/gtx/string_cast.hpp>

#include <stdio.h>

void PointCloud::addPoint(const SphereMesh & sphereMesh)
{
    //printf("[METHOD] addPoint...\n");
    int dimensionality = -1;
    glm::vec3 pos;
    Point point;
    while (true) {
        pos = getRandomPositionInSphere(sphereMesh.boundingSphere);
        //printf("Got random position in sphere: %s\n", glm::to_string(pos).c_str());
        point = sphereMesh.pushOutside(pos, dimensionality);
        //printf("Points pushed outside by %d\n", dimensionality);
        //point is inside the sphere mesh and has been pushed on its surface
        if (dimensionality != -1) {
            points.emplace_back(point.pos, point.normal, dimensionality);
            return;
        }
    }
}

unsigned int PointCloud::getPointsNumber() const {
    return points.size();
}



void PointCloud::repopulate(const unsigned int nPoints, const SphereMesh& sphereMesh) {
    //if we are asking for less than the actual number of points, erase the surplus
    //else create points.size() - nPoints new points
    if (nPoints < points.size()) {
        points.erase(points.end() - (points.size() - nPoints), points.end());
        return;
    }

    for (size_t i = 0; i < nPoints - points.size(); i++) {
        addPoint(sphereMesh);
    }
}

void PointCloud::clear() {
    points.clear();
}

const void* PointCloud::pointerToData() const {
    return points.data();
}
