#include <utils/pointcloud.h>
#include <spheremeshes/sphere.h>
#include <spheremeshes/spheremesh.h>

//TODO temporary, removed this
static const glm::vec3 COLORS[3] = {glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)};

//TODO IMPO: continua da qui
void PointCloud::addPoint(const SphereMesh & sphereMesh)
{
    int dimensionality = -1;
    glm::vec3 pos;
    Point point;
    while (true) {
        pos = getRandomPositionInSphere(sphereMesh.boundingSphere);
        point = sphereMesh.pushOutside(pos, dimensionality);
        //point is inside the sphere mesh and has been pushed on its surface
        if (dimensionality != -1) {
            points.emplace_back(point, COLORS[dimensionality]);
            return;
        }
    }
}
