#pragma once

#include <glm/glm.hpp>
#include <vector>

typedef unsigned int uint;

//check if n is in range [min, max] inclusive
bool isInRangeIncl(float n, float min, float max);

//check if n is in range (min, max) exclusive
bool isInRangeExcl(float n, float min, float max);

//compute the direction in world space from a position on the screen
glm::vec3 screenToWorldDir(const glm::vec2& screenPos, float width, float height, const glm::mat4& viewMatrix, const glm::mat4 projectionMatrix);

void triangulateSquareGrid(uint dim, std::vector<uint>& outIndices);

uint linearizedIndexSquareGrid(uint dim, uint x, uint y);