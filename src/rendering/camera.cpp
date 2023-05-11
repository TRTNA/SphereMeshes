#include <rendering/camera.h>
#include <cassert>

#include <glm/gtc/matrix_transform.hpp>

using glm::mat4;
using glm::vec3;

void Camera::updateViewMatrix()
{
    viewMatrix = glm::lookAt(pos, forward, up);
}

void Camera::updateProjectionMatrix()
{
    assert(height > 0.0f);
    projectionMatrix = glm::perspective(fovY, width/height, near, far);
}

void Camera::translate(glm::vec3 t)
{
    pos += t;
    updateViewMatrix();
}
void Camera::setPos(glm::vec3 newPos)
{
    pos = newPos;
    updateViewMatrix();
}

void Camera::setForward(glm::vec3 forward)
{
    this->forward = forward;
    updateViewMatrix();
}
void Camera::setNearPlane(float near)
{
    this->near = near;
    updateProjectionMatrix();
}

void Camera::setFarPlane(float far)
{
    this->far = far;
    updateProjectionMatrix();
}
void Camera::setFrameWidth(float width)
{
    this->width = width;
    updateProjectionMatrix();
}
void Camera::setFrameHeight(float height)
{
    this->height = height;
    updateProjectionMatrix();
}
void Camera::setFovY(float fovY)
{
    this->fovY = fovY;
    updateProjectionMatrix();
}

glm::vec3 Camera::getPos() const
{
    return pos;
}
glm::vec3 Camera::getForward() const
{
    return forward;
}
float Camera::getNearPlane() const
{
    return near;
}
float Camera::getFarPlane() const
{
    return far;
}
float Camera::getFrameWidth() const
{
    return width;
}
float Camera::getFrameHeight() const
{
    return height;
}
float Camera::getFovY() const
{
    return fovY;
}
glm::mat4 Camera::getProjectionMatrix() const
{
    return projectionMatrix;
}
glm::mat4 Camera::getViewMatrix() const
{
    return viewMatrix;
}