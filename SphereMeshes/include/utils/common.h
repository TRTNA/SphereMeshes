#pragma once

#include <glm/glm.hpp>
#include <vector>

struct Sphere;

typedef unsigned int uint;

//check if n is in range [min, max] inclusive
bool isInRangeIncl(float n, float min, float max);

//check if n is in range (min, max) exclusive
bool isInRangeExcl(float n, float min, float max);

//compute the direction in world space from a position on the screen
glm::vec3 screenToWorldDir(const glm::vec2& screenPos, float width, float height, const glm::mat4& viewMatrix, const glm::mat4 projectionMatrix);

void triangulateSquareGrid(uint dim, std::vector<uint>& outIndices);

uint linearizedIndexSquareGrid(uint dim, uint x, uint y);

float computeVolume(const Sphere& sphere);


glm::mat3 fromToRotate(glm::vec3 from, glm::vec3 to);

glm::mat3 fromToRotate(glm::vec3 from1, glm::vec3 to1, glm::vec3 from2, glm::vec3 to2);
