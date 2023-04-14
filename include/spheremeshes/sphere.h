#pragma once

#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <ostream>

struct Sphere {
    Sphere();
    Sphere(glm::vec3 pCenter, float pRadius);
    glm::vec3 center;
    float radius;
};

std::ostream& operator<<(std::ostream& ost, const Sphere& val);

const Sphere& getBiggerSphere(const Sphere& s1, const Sphere& s2);

//TODO
// metodo che date due sfere restituisce la sfera più piccola che le contenga entrambe
Sphere computeBoundingSphere(const Sphere& s1, const Sphere& s2);

//TODO
// metodo che restituisce punto casuale nella sfera
// punto a caso nel cubo che contiene la sfera (3 numeri random x,y,z)
// se è fuori dalla sfera, lo scarto e reitero finchè non è dentro.
