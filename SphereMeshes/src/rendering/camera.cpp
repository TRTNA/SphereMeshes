#include <rendering/camera.h>
#include <cassert>

#include <glm/gtc/matrix_transform.hpp>

using namespace glm;

Camera::Camera(glm::vec3 pos, glm::vec3 forward, float near, float far, float width, float height, float fovY) 
: pos(pos), forward(forward), up(vec3(0.0f, 1.0f, 0.0f)), near(near), far(far), width(width), height(height), fovY(fovY) 
{
    updateViewMatrix();
    updateProjectionMatrix();
}


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


void Camera::rotateAroundY(float angleDeg) {
    mat4 rot(1.0f);
    rot = glm::rotate(rot, angleDeg, glm::vec3(0.0f, 1.0f, 0.0f));
    pos = vec3(rot * vec4(pos, 1.0f));
    forward = -pos;
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