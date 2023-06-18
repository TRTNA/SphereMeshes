#include <spheremeshes/sphere.h>
#include <glm/common.hpp>
#include <glm/gtx/string_cast.hpp>
#include <spheremeshes/point.h>
#include <utils/random.h>
#include <utils/aabb.h>

#include <utils/types.h>

Sphere::Sphere(glm::vec3 pCenter, float pRadius) : center(pCenter), radius(pRadius) {}

Sphere::Sphere() : center(glm::vec3(0.0f)), radius(0.0f) {}

void Sphere::scale(float k) {
    center *= k;
    radius *= k;
}

std::ostream& operator<<(std::ostream& ost, const Sphere& val) {
    ost << val.center.x << " " << val.center.y << " " << val.center.z << " " << val.radius;
    return ost;
}

std::istream& operator>>(std::istream& ist, Sphere& val) {
    ist >> val.center.x >> val.center.y >> val.center.z >> val.radius;
    return ist;
}


const Sphere& getBiggerSphere(const Sphere& s1, const Sphere& s2) {
    return s1.radius > s2.radius ?  s1 : s2;
}

bool isInside(const glm::vec3& pos, const Sphere& sphere) {
    glm::vec3 posToCenter = sphere.center - pos;
    //sqrd distance from pos to sphere center is less than squared radius --> inside
    return dot(posToCenter, posToCenter) < sphere.radius * sphere.radius + EPSILON;
}



 
Sphere computeBoundingSphere(const Sphere& s1, const Sphere& s2)
{
    glm::vec3 c2c1 = s2.center - s1.center;
    float c2c1mag = glm::length(c2c1);
    // if the distance between the center is less than the greatest radius then the smaller sphere is inside the bigger sphere, so bounding sphere is the bigger.
    if (c2c1mag <= glm::abs(s1.radius - s2.radius)) {
        return Sphere(getBiggerSphere(s1, s2));
    }
    float R = (s2.radius + s1.radius + c2c1mag) / 2.0f;
    glm::vec3 C = s1.center + c2c1*(R - s1.radius) / c2c1mag;
    return Sphere(C, R);
}

glm::vec3 getRandomPositionInSphere(const Sphere& sphere) {
    Point point;
    const AABB& sphereAABB = computeAABB(sphere);
    //infinite loop but it's okay, high probability of having a point inside the sphere in some iterations
    while (true) {
        glm::vec3 pointPos = generatePosition(sphereAABB);
        glm::vec3 centerToPos = pointPos - sphere.center;
        //generated position is inside the sphere, so return the Point with position pointPos
        if (glm::dot(centerToPos, centerToPos) <= sphere.radius*sphere.radius) {
            return pointPos;
        }
    }
}

Sphere computeBoundingSphere(std::vector<Sphere> spheres) {
    if (spheres.size() == 0) return Sphere();
    if (spheres.size() == 1) return spheres.at(0);
    Sphere& boundingSphere = spheres.at(0);
    for (size_t i = 1; i < spheres.size(); i++) {
        boundingSphere = computeBoundingSphere(boundingSphere, spheres.at(i));
    }
    return boundingSphere;
}

AABB computeAABB(const Sphere& sphere) {
    return AABB(
        floatRange(sphere.center.x - sphere.radius, sphere.center.x + sphere.radius),
        floatRange(sphere.center.y - sphere.radius, sphere.center.y + sphere.radius),
        floatRange(sphere.center.z - sphere.radius, sphere.center.z + sphere.radius)
    );
}

