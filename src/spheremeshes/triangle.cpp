#include <spheremeshes/triangle.h>

#include <string>

using std::array;

Triangle::Triangle(uint v0, uint v1, uint v2) : vertices(array<uint, 3>{{v0, v1, v2}}) {}
Triangle::Triangle(const std::array<uint, 3>& pVertices) : vertices(array<uint, 3>{pVertices}) {}

std::ostream& operator<<(std::ostream& ost, const Triangle& val) {
    ost << val.vertices.at(0) << " ";
    ost << val.vertices.at(1) << " ";
    ost << val.vertices.at(2);
    return ost;
}

std::istream& operator>>(std::istream& ist, Triangle& val) {
    ist >> val.vertices.at(0) >> val.vertices.at(1) >> val.vertices.at(2);
    return ist;
}

