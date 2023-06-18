#include <utils/common.h>

#include <spheremeshes/sphere.h>
#include <physics/physicsconstants.h>
#include <glm/gtx/vector_angle.hpp>

#include <spheremeshes/spheremesh.h>

using std::vector;
using glm::vec3;
using glm::vec4;

bool isInRangeIncl(float n, float min, float max) {
    return n >= min && n <= max;
}

bool isInRangeExcl(float n, float min, float max) {
    return n > min && n < max;
}

glm::vec3 screenToWorldDir(const glm::vec2& screenPos, float width, float height, const glm::mat4& viewMatrix, const glm::mat4 projectionMatrix) {
        float x = (2.0f * screenPos.x) / width - 1.0f;
        float y = 1.0f - (2.0f * screenPos.y) / height;
        float z = 1.0f;
        vec3 ray_nds = vec3(x, y, z);
        vec4 ray_clip = vec4(ray_nds.x, ray_nds.y, -1.0, 1.0);
        vec4 ray_eye = glm::inverse(projectionMatrix) * ray_clip;
        ray_eye = vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);
        vec3 ray_wor = glm::vec3((glm::inverse(viewMatrix) * ray_eye));
        return glm::normalize(ray_wor);
}

void triangulateSquareGrid(uint dim, std::vector<uint>& outIndices) {
    for (size_t i = 0; i < dim - 1; i++)
    {
        for (size_t j = 0; j < dim - 1; j++)
        {
            outIndices.push_back(dim * i + j);
            outIndices.push_back(dim * i + j + 1);
            outIndices.push_back(dim * (i + 1) + j);

            outIndices.push_back(dim * (i + 1) + j);
            outIndices.push_back(dim * i + j + 1);
            outIndices.push_back(dim * (i + 1) + j + 1);
        }
    }
}

uint linearizedIndexSquareGrid(uint dim, uint x, uint y) {
    return x * dim + y;
}


float computeVolume(const Sphere& sphere) {
    return 4.0f/3.0f * 3.1415926535f * sphere.radius;
}

glm::mat3 fromToRotate(glm::vec3 from, glm::vec3 to) {
    glm::vec3 axis = glm::cross(from, to);
    if (glm::dot(axis, axis) < EPSILON) return glm::mat3(1.0f);
    float angle = glm::angle(glm::normalize(from), glm::normalize(to));
    return glm::mat3(glm::rotate(angle, axis));
}

glm::mat3 fromToRotate(glm::vec3 from1, glm::vec3 to1, glm::vec3 from2, glm::vec3 to2) {
    glm::mat3 rotMatrixA = fromToRotate(from1, to1);
    from2 = rotMatrixA * from2;
    to2 = glm::cross(to1, glm::cross(to2, to1));
    from2 = glm::cross(to1, glm::cross(from2, to1));
    glm::mat3 rotMatrixB = glm::rotate(glm::angle(glm::normalize(from2), glm::normalize(to2)), to1);
    return rotMatrixB * rotMatrixA;
}
