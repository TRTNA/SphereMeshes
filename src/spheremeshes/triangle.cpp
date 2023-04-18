#include <spheremeshes/triangle.h>

#include <string>

using std::array;

Triangle::Triangle(U32 v0, U32 v1, U32 v2) : vertices(array<U32, 3>{{v0, v1, v2}}) {}
Triangle::Triangle(const std::array<U32, 3>& pVertices) : vertices(array<U32, 3>{pVertices}) {}

std::ostream& operator<<(std::ostream& ost, const Triangle& val) {
    ost << "Triangle(";
    ost << std::to_string(val.vertices.at(0)) << " ";
    ost << std::to_string(val.vertices.at(1)) << " ";
    ost << std::to_string(val.vertices.at(2)) << ")";
    return ost;
}

std::istream& operator>>(std::istream& ist, Triangle& val) {
    ist >> val.vertices.at(0) >> val.vertices.at(1) >> val.vertices.at(2);
    return ist;
}

