#include <spheremeshes/edge.h>
#include <string>

std::ostream& operator<<(std::ostream& ost, const Edge& val) {
    ost << val.first << " ";
    ost << val.second;
    return ost;

}

std::istream& operator>>(std::istream& ist, Edge& val) {
    ist >> val.first >> val.second;
    return ist;
}

