#ifndef _SPHERE_H
#define _SPHERE_H

#include <glm/vec3.hpp>
#include <ostream>

struct Sphere {
    Sphere();
    Sphere(glm::vec3 pCenter, float pRadius);
    glm::vec3 center;
    float radius;
};

std::ostream& operator<<(std::ostream& ost, const Sphere& val);

//TODO
// metodo che date due sfere restituisce la sfera più piccola che le contenga entrambe

//TODO
// metodo che restituisce punto casuale nella sfera
// punto a caso nel cubo che contiene la sfera (3 numeri random x,y,z)
// se è fuori dalla sfera, lo scarto e reitero finchè non è dentro.

#endif