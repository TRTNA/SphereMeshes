#include <spheremeshes/spheremesh.h>

#include <glm/vec3.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <fstream>

using glm::vec3;
using std::ifstream;
using std::ostream;
using std::string;


void updateCapsuloidFeatures(Capsuloid& caps, const Sphere& s0, const Sphere& s1)
{
    caps.S0toS1 = s1.center - s0.center;
    caps.sqrdL = glm::dot(caps.S0toS1, caps.S0toS1);
    caps.factor = (s1.radius - s0.radius) / caps.sqrdL;
}

void updateSphereTriangleFeatures(SphereTriangle& tri, const Sphere& s0, const Sphere& s1, const Sphere& s2)
{
    tri.S0S1 = s1.center - s0.center;
    tri.S0S2 = s2.center - s0.center;

    tri.planeN = glm::normalize(glm::cross(tri.S0S1, tri.S0S2));
    glm::vec3 upperPlaneN = tri.planeN;
    glm::vec3 lowerPlaneN = - tri.planeN;

    const vec3 C0minusC1 = -tri.S0S1;
    const vec3 C2minusC1 = s2.center - s1.center;
    vec3 e = vec3(1.0f);
    while (e.x > EPSILON && e.y > EPSILON && e.z > EPSILON)
    {
        glm::mat3 A = glm::rowMajor3(C0minusC1, C2minusC1, upperPlaneN);
        vec3 t = vec3(
            s1.radius - s0.radius - glm::dot(C0minusC1, upperPlaneN),
            s1.radius - s2.radius - glm::dot(C2minusC1, upperPlaneN),
            0.0f);
        e = glm::inverse(A) * t;
        upperPlaneN = glm::normalize(upperPlaneN + e);
        lowerPlaneN = glm::normalize(lowerPlaneN + e);
    } 

    tri.upperProjMatrix = glm::inverse(glm::mat3(tri.S0S1, tri.S0S2, upperPlaneN));
    tri.lowerProjMatrix = glm::inverse(glm::mat3(tri.S0S1, tri.S0S2, lowerPlaneN));

}


void toSphereTriangleReferenceSystem(const SphereTriangle& tri, const glm::vec3& q, float& outA, float& outB, float& outC, float& outD) {
    const glm::mat3 projMatrix = glm::dot(q, tri.planeN) < 0 ? tri.lowerProjMatrix : tri.upperProjMatrix;
    const vec3 res = projMatrix * q;
    outD = res.z;
    outA = res.x;
    outB = res.y;
    outC = (1.0f - outA - outB);
}

bool readFromFile(const std::string &path, SphereMesh &out, std::string &errorMsg)
{
    ifstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        errorMsg = "Cannot open file " + path;
        return false;
    }

    try
    {
        uint spheresNo = 0;
        file >> spheresNo;
        for (size_t i = 0; i < spheresNo; i++)
        {
            Sphere sphere;
            file >> sphere;
            out.addSphere(sphere);
        }

        uint singletonsNo = 0;
        file >> singletonsNo;
        for (size_t i = 0; i < singletonsNo; i++)
        {
            uint sphereIdx;
            file >> sphereIdx;
            out.addSingleton(sphereIdx);
        }

        uint capsNo = 0;
        file >> capsNo;
        for (size_t i = 0; i < capsNo; i++)
        {
            Capsuloid caps;
            file >> caps;
            out.addCapsuloid(caps);
        }

        uint sphereTrianglesNo = 0;
        file >> sphereTrianglesNo;
        for (size_t i = 0; i < sphereTrianglesNo; i++)
        {
            SphereTriangle sphereTriangle;
            file >> sphereTriangle;
            out.addSphereTriangle(sphereTriangle);
        }
    }
    catch (const std::ios::failure &ex)
    {
        errorMsg = "Error while reading " + path +"\n";
        file.close();
        return false;
    }

    out.updateBoundingSphere();
    file.close();
    return true;
}

std::ostream &operator<<(std::ostream &ost, const SphereMesh &sm)
{
    ost << sm.spheres.size() << "\n";
    for (const auto &s : sm.spheres)
    {
        ost << s << "\n";
    }
    ost << "\n";
    ost << sm.singletons.size() << "\n";
    for (const auto &s : sm.singletons)
    {
        ost << s << "\n";
    }
    ost << "\n";
    ost << sm.capsuloids.size() << "\n";
    for (const auto &e : sm.capsuloids)
    {
        ost << e << "\n";
    }
    ost << "\n";
    ost << sm.sphereTriangles.size() << "\n";
    for (const auto &t : sm.sphereTriangles)
    {
        ost << t << "\n";
    }
    return ost;
}


Point pointOutsideSphereMesh(const glm::vec3& pos, int &dimensionality) {
    dimensionality = -1;
    return Point(pos, glm::vec3(0.0f));
}


