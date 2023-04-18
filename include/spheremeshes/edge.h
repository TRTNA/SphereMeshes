#ifndef _EDGE_H
#define _EDGE_H

#include <utils/types.h>
#include <iostream>


typedef std::pair<uint, uint> Edge;
std::ostream& operator<<(std::ostream& ost, const Edge& val);
std::istream& operator>>(std::istream& ost, Edge& val);

#endif