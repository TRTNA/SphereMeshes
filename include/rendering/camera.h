#pragma once

class Camera {
    private:
        glm::vec3 pos = glm::vec3(0.0f);
        glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        glm::mat4 viewMatrix;
        glm::mat4 projectionMatrix;
    public:
        void translate(glm::vec3 t);
        void setPos(glm::vec3 newPos);
        void setForward(glm::vec3 forward);
        void setNearPlane(float near);
        void setFarPlane(float far);
        void setFrameWidth(float width);
        void setFrameHeight(float height);
        void setFovY(float fovY);

        glm::vec3 getPos();
        glm::vec3 getForward();
        float getNearPlane() const;
        float getFarPlane() const;
        float getFrameWidth() const;
        float getFrameHeight() const;
        float getFovY() const;
        glm::mat4 getProjectionMatrix() const;
        glm::mat4 getViewMatrix() const;
};