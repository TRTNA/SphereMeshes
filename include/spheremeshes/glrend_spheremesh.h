#ifndef _GLREND_SPHEREMESH_H
#define _GLREND_SPHEREMESH_H

#include <utils/iglrenderable.h>
#include <utils/shader.h>
#include <utils/model.h>
#include <spheremeshes/sphere.h>
#include <spheremeshes/edge.h>
#include <spheremeshes/triangle.h>
#include <spheremeshes/spheremesh.h>
#include <utils/pointcloud.h>


#include <vector>

class GlRendSphereMesh : public IglRenderable, SphereMesh {
    private:
        unsigned int VAO, VBO;
        std::vector<PointCloud> pcs;
    public:
        static Model* sphereModel;
        GlRendSphereMesh() = default;
        GlRendSphereMesh(std::vector<Sphere>& pSpheres, std::vector<Edge>& pEdges, std::vector<Triangle>& pTriangles);
        void Draw(const Shader& shader) override;
        ~GlRendSphereMesh();
};

#endif