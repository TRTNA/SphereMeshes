#include <spheremeshes/SphereTriangle.h>

#include <string>

using std::array;

SphereTriangle::SphereTriangle(uint v0, uint v1, uint v2) : s0(v0), s1(v1), s2(v2) {}

SphereTriangle::SphereTriangle(const std::array<uint, 3>& pVertices) : s0(pVertices[0]), s1(pVertices[1]), s2(pVertices[2]) {}


std::ostream& operator<<(std::ostream& ost, const SphereTriangle& val) {
    ost << val.s0 << " ";
    ost << val.s1 << " ";
    ost << val.s2;
    return ost;
}

std::istream& operator>>(std::istream& ist, SphereTriangle& val) {
    ist >> val.s0 >> val.s1 >> val.s2;
    return ist;
}



