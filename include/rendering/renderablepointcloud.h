#pragma once

#include <rendering/iglrenderable.h>
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
        void draw() override;
        void updateBuffers();
};