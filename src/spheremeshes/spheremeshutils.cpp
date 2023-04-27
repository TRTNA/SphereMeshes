#include <spheremeshes/spheremesh.h>

#include <glm/vec3.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <sstream>
#include <fstream>
using glm::vec3;
using std::ifstream;
using std::ostream;
using std::string;
using std::stringstream;

bool readFromFile(const std::string &path, SphereMesh &out, std::string &errorMsg)
{
    ifstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path, std::ios::in);
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
        errorMsg = path + " badly formattated\n";
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

float computeCapsuloidFactor(const Sphere &s0, const Sphere &s1)
{
    const vec3 l = s1.center - s0.center;
    return (s1.radius - s0.radius) / (glm::dot(l, l));
}

void SphereMesh::updateSphereTriangleFeatures(SphereTriangle &tri)
{
    const Sphere &s0 = spheres.at(tri.vertices[0]);
    const Sphere &s1 = spheres.at(tri.vertices[1]);
    const Sphere &s2 = spheres.at(tri.vertices[2]);

    tri.S0S1 = s1.center - s0.center;
    tri.S0S2 = s2.center - s0.center;

    tri.planeN = glm::normalize(glm::cross(tri.S0S1, tri.S0S2));
    tri.upperPlaneN = tri.planeN;
    tri.bottomPlaneN = - tri.planeN;

    const vec3 C0minusC1 = -tri.S0S1;
    const vec3 C2minusC1 = s2.center - s1.center;
    vec3 e = vec3(1.0f, 1.0f, 1.0f);
    do
    {
        glm::mat3 A = glm::rowMajor3(C0minusC1, C2minusC1, tri.upperPlaneN);
        vec3 t = vec3(
            s1.radius - s0.radius - glm::dot(C0minusC1, tri.upperPlaneN),
            s1.radius - s2.radius - glm::dot(C2minusC1, tri.upperPlaneN),
            0.0f);
        e = glm::inverse(A) * t;
        tri.upperPlaneN = glm::normalize(tri.upperPlaneN + e);
        tri.bottomPlaneN = glm::normalize(tri.bottomPlaneN + e);
    } while (e.x > EPSILON && e.y > EPSILON && e.z > EPSILON);
}

void SphereMesh::scale(float k)
{
    for (auto &s : spheres)
    {
        s.scale(k);
    }
    boundingSphere.scale(k);
    updateAllCapsuloidsFactors();
    updateAllSphereTriangleFeatures();
}

void toSphereTriangleReferenceSystem(const SphereTriangle& tri, const glm::vec3& q, float& outA, float& outB, float& outC, float& outD) {
    float d, k0, k1, a, b, c;
    glm::mat3 projMatrix;
    if (glm::dot(q, tri.planeN) < 0)
    {
        projMatrix = glm::inverse(glm::mat3(tri.S0S2, tri.S0S1, tri.bottomPlaneN));
        vec3 res = projMatrix * q;
        outD = res.z;
        k0 = res.y;
        k1 = res.x;
        outA = k0;
        outB = k1;
        outC = (1.0f - k0 - k1);
    }
    else
    {
        projMatrix = glm::inverse(glm::mat3(tri.S0S1, tri.S0S2, tri.upperPlaneN));
        vec3 res = projMatrix * q;

        d = res.z;
        k0 = res.x;
        k1 = res.y;

        outA = k0;
        outB = k1;
        outC = (1.0f - k0 - k1);
    }

}

Point pointOutsideSphereMesh(const glm::vec3& pos, int &dimensionality) {
    dimensionality = -1;
    return Point(pos, glm::vec3(0.0f));
}


