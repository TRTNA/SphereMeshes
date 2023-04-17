#pragma once

#include <utils/iglrenderable.h>
#include <utils/pointcloud.h>
#include <memory>

class RenderablePointCloud : public IglRenderable {
    private:
        unsigned int VAO, VBO;
        unsigned int pointsNumber;
        std::shared_ptr<PointCloud> pointCloud;
    public:
        RenderablePointCloud(std::shared_ptr<PointCloud> ptr);
        ~RenderablePointCloud();
        void Draw(const Shader& shader) override;
        void updateBuffers();
};