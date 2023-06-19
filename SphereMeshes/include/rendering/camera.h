#pragma once

#include <glm/glm.hpp>

class Camera {
    private:
        glm::vec3 pos = glm::vec3(0.0f);
        glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        float near = 0.0f, far = 0.0f;
        float width = 1.0f, height = 1.0f;
        float fovY = 0.0f;
        glm::mat4 viewMatrix;
        glm::mat4 projectionMatrix;
        void updateViewMatrix();
        void updateProjectionMatrix();
    public:
        Camera() = default;
        Camera(glm::vec3 pos, glm::vec3 forward, float near, float far, float width, float height, float fovY);

        void translate(glm::vec3 t);
        void rotateAroundY(float angleDeg);

        void setPos(glm::vec3 newPos);
        void setForward(glm::vec3 forward);
        void setNearPlane(float near);
        void setFarPlane(float far);
        void setFrameWidth(float width);
        void setFrameHeight(float height);
        void setFovY(float fovY);

        glm::vec3 getPos() const;
        glm::vec3 getForward() const;
        float getNearPlane() const;
        float getFarPlane() const;
        float getFrameWidth() const;
        float getFrameHeight() const;
        float getFovY() const;
        glm::mat4 getProjectionMatrix() const;
        glm::mat4 getViewMatrix() const;
};