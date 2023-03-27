#include <spheremeshes/edge.h>
#include <string>

std::ostream& operator<<(std::ostream& ost, const Edge& val) {
    ost << "Edge(";
    ost << std::to_string(val.first) << " ";
    ost << std::to_string(val.second) << ")";
    return ost;

}
