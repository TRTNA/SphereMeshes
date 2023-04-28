#pragma once

#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <iostream>
#include <vector>

using std::vector;

struct Point;
class AABB;

/*
    Simple sphere with a given center and radius
*/
struct Sphere {
    Sphere();
    Sphere(glm::vec3 pCenter, float pRadius);
    void scale(float k);
    glm::vec3 center;
    float radius;
};

std::ostream& operator<<(std::ostream& ost, const Sphere& val);
std::istream& operator>>(std::istream& ist, Sphere& val);

bool isInside(const glm::vec3& pos, const Sphere& sphere);

const Sphere& getBiggerSphere(const Sphere& s1, const Sphere& s2);

Sphere computeBoundingSphere(const Sphere& s1, const Sphere& s2);

Sphere computeBoundingSphere(std::vector<Sphere> spheres);

glm::vec3 getRandomPositionInSphere(const Sphere& sphere);

AABB computeAABB(const Sphere& sphere);
