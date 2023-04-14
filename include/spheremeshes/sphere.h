#pragma once

#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <ostream>
#include <vector>

using std::vector;

struct Point;
class AABB;

struct Sphere {
    Sphere();
    Sphere(glm::vec3 pCenter, float pRadius);
    glm::vec3 center;
    float radius;
};

std::ostream& operator<<(std::ostream& ost, const Sphere& val);

const Sphere& getBiggerSphere(const Sphere& s1, const Sphere& s2);

Sphere computeBoundingSphere(const Sphere& s1, const Sphere& s2);

//TODO DA TESTARE FORSE NON FUNZA
Sphere computeBoundingSphere(std::vector<Sphere> spheres);

Point getRandomPointInSphere(const Sphere& sphere);

AABB computeAABB(const Sphere& sphere);
