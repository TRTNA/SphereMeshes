#include <spheremeshes/point.h>

 Point::Point() : Point::Point(glm::vec3(0.0f), glm::vec3(0.0f)) {}

 Point::Point(glm::vec3 pos, glm::vec3 normal) : pos(pos), normal(normal) {}


DimensionalityPoint::DimensionalityPoint() : DimensionalityPoint(glm::vec3(0.0f), glm::vec3(0.0f), -1) {}
DimensionalityPoint::DimensionalityPoint(glm::vec3 pos, glm::vec3 normal, int pDimensionality) : pos(pos), normal(normal), dimensionality(pDimensionality) {}


 ColoredPoint::ColoredPoint() : Point(), color(glm::vec3(0.0f))
 {
 }

 ColoredPoint::ColoredPoint(glm::vec3 pos, glm::vec3 normal, glm::vec3 color) : Point(pos, normal), color(color)
 {
 }

 ColoredPoint::ColoredPoint(const Point &point, glm::vec3 color) : Point(point), color(color)
 {
 }
