#include <spheremeshes/sphere.h>
#include <glm/common.hpp>
#include <glm/gtx/string_cast.hpp>
#include <spheremeshes/point.h>
#include <utils/random.h>
#include <utils/aabb.h>

Sphere::Sphere(glm::vec3 pCenter, float pRadius) : center(pCenter), radius(pRadius) {}

Sphere::Sphere() : center(glm::vec3(0.0f)), radius(0.0f) {}

std::ostream& operator<<(std::ostream& ost, const Sphere& val) {
    ost << "Sphere(";
    ost << glm::to_string(val.center);
    ost << "; ";
    ost << val.radius << ")";
    return ost;
}

const Sphere& getBiggerSphere(const Sphere& s1, const Sphere& s2) {
    return s1.radius > s2.radius ?  s1 : s2;
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

Point getRandomPointInSphere(const Sphere& sphere) {
    Point point;
    const AABB& sphereAABB = computeAABB(sphere);
    //infinite loop but it's okay, high probability of having a point inside the sphere in some iterations
    while (true) {
        glm::vec3 pointPos = generatePosition(sphereAABB);
        glm::vec3 centerToPos = pointPos - sphere.center;
        //generated position is inside the sphere, so return the Point with position pointPos
        if (glm::dot(centerToPos, centerToPos) < sphere.radius*sphere.radius) {
            return Point(pointPos, glm::vec3(0.0f));
        }
    }
}

AABB computeAABB(const Sphere& sphere) {
    return AABB(
        floatRange(sphere.center.x - sphere.radius, sphere.center.x + sphere.radius),
        floatRange(sphere.center.y - sphere.radius, sphere.center.y + sphere.radius),
        floatRange(sphere.center.z - sphere.radius, sphere.center.z + sphere.radius)
    );
}

