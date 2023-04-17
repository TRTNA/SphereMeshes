#include <spheremeshes/sphere.h>

#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <cassert>


using namespace std;

int main()  {
    cout << "Started..." << endl;
    Sphere s1(glm::vec3(0.0f), 1.0f);
    Sphere s2(glm::vec3(3.0f, 0.0f, 0.0f), 2.0f);
    Sphere s3(glm::vec3(-2.0f), 1.0f);
    Sphere b = computeBoundingSphere(std::vector<Sphere> {s1, s2, s3});
    cout << glm::to_string(b.center) << " " << b.radius << endl;
}