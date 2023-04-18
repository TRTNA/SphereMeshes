#ifndef _EDGE_H
#define _EDGE_H

#include <utils/types.h>
#include <ostream>


typedef std::pair<U32, U32> Edge;
std::ostream& operator<<(std::ostream& ost, const Edge& val);
std::istream& operator<<(std::istream& ost, Edge& val);

#endif