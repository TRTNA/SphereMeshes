#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include <utils/types.h>
#include <array>
#include <ostream>

struct Triangle {
    std::array<U32, 3> vertices;
    Triangle() = default;
    Triangle(uint v0, uint v1, uint v2);
    Triangle(const std::array<uint, 3>& pVertices);
};

std::ostream& operator<<(std::ostream& ost, const Triangle& val);
std::istream& operator>>(std::istream& ist, Triangle& val);

#endif