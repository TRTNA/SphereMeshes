#pragma once

#include <utils/iglrenderable.h>
#include <utils/shader.h>
#include <utils/model.h>
#include <utils/pointcloud.h>

#include <spheremeshes/sphere.h>
#include <spheremeshes/edge.h>
#include <spheremeshes/triangle.h>
#include <spheremeshes/spheremesh.h>


#include <memory>
#include <vector>

static const unsigned int DEFAULT_POINTS_NUMBER = 10000;

class GlRendSphereMesh : public IglRenderable, SphereMesh {
    private:
        unsigned int VAO, VBO;
        unsigned int pointsNumber;
        std::vector<std::shared_ptr<PointCloud>> pcs;
    public:
        GlRendSphereMesh(unsigned int pointsNumber = DEFAULT_POINTS_NUMBER);
        GlRendSphereMesh(std::vector<Sphere>& pSpheres, std::vector<Edge>& pEdges, std::vector<Triangle>& pTriangles, unsigned int pointsNumber = DEFAULT_POINTS_NUMBER);
        void Draw(const Shader& shader) override;
        void setPointsNumber(unsigned int pPointsNumber);
        void regeneratePoints();
        
        ~GlRendSphereMesh();
};
