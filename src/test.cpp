#include <cloth/cloth.h>

#include <glm/gtx/string_cast.hpp>
#include <glm/glm.hpp>

#include <iostream>
#include <cassert>

using glm::vec3;

using namespace std;

int main()  {
    cout << "Started..." << endl;
    Cloth cloth(4, 5);
    cout << cloth.toString();

    vec3** pts;
    cloth.getPoints(pts);
    pts[0][0] = vec3(-2.0f, -2.0f, 0.0f);
    cloth.enforceConstraints();

    cout << cloth.toString();
}