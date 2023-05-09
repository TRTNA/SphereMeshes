#include <utils/common.h>

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
            outIndices.push_back(dim * (i + 1) + j);
            outIndices.push_back(dim * (i + 1) + j + 1);

            outIndices.push_back(dim * i + j);
            outIndices.push_back(dim * (i + 1) + j + 1);
            outIndices.push_back(dim * i + j + 1);
        }
    }
}