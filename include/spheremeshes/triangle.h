#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include <utils/types.h>
#include <array>
#include <ostream>

struct Triangle {
    std::array<U32, 3> vertices;
    Triangle(U32 v0, U32 v1, U32 v2);
    Triangle(const std::array<U32, 3>& pVertices);
};

std::ostream& operator<<(std::ostream& ost, const Triangle& val);

#endif