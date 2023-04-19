#include <spheremeshes/spheremesh.h>

#include <iostream>
#include <stdio.h>
#include <sstream>
#include <array>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

using std::clog;
using std::endl;

using std::array;
using std::ifstream;
using std::ostream;
using std::stringstream;
using std::vector;
using std::string;

static const float EPSILON = 0.001f;

SphereMesh::SphereMesh(vector<Sphere> &pSpheres, vector<Capsuloid> &pCapsuloids, vector<Triangle> &pTriangles, vector<uint> &pSingletons)
    : spheres(std::move(pSpheres)), capsuloids(std::move(pCapsuloids)), triangles(std::move(pTriangles)), singletons(std::move(pSingletons))
{
    clog << "Created a sphere mesh:\n";
    clog << "- Spheres:\t" << spheres.size() << "\n";
    clog << "- Capsuloids:\t" << capsuloids.size() << "\n";
    clog << "- Triangles:\t" << triangles.size() << "\n";
    clog << "- Singletons:\t" << singletons.size() << "\n";
    updateBoundingSphere();
}

void SphereMesh::addSphere(const Sphere &sphere)
{
    spheres.emplace_back(sphere.center, sphere.radius);
}

void SphereMesh::addCapsuloid(const Capsuloid &caps)
{
    capsuloids.emplace_back(caps.s0, caps.s1);
}

void SphereMesh::addTriangle(const Triangle &triangle)
{
    triangles.emplace_back(triangle.vertices);
}

void SphereMesh::addSingleton(uint sphereIdx)
{
    singletons.push_back(sphereIdx);
}

void SphereMesh::updateBoundingSphere()
{
    boundingSphere = computeBoundingSphere(spheres);
}
std::string SphereMesh::toString() const
{
    stringstream ss;
    ss << "Spheres:\n";
    for (size_t i = 0; i < spheres.size(); i++)
    {
        ss << i << " " << spheres.at(i) << "\n";
    }
    ss << "\n";
    ss << "Edges:\n";
    for (size_t i = 0; i < capsuloids.size(); i++)
    {
        ss << i << " " << capsuloids.at(i) << "\n";
    }
    ss << "\n";
    ss << "Triangles:\n";
    for (size_t i = 0; i < triangles.size(); i++)
    {
        ss << i << " " << triangles.at(i) << "\n";
    }
    ss << "\n";
    ss << "Singletons:\n";
    for (size_t i = 0; i < singletons.size(); i++)
    {
        ss << i << " " << singletons.at(i) << "\n";
    }
    return ss.str();
}
Point SphereMesh::pushOutside(const glm::vec3 &pos, int &dimensionality) const
{
    bool outsideEverything = false;
    Point lastPoint = Point(pos, glm::vec3(0.0f));
    int lastDimensionality = -1;
    uint singletonStart = 0;
    uint edgeStart = singletonStart + singletons.size();
    uint triangleStart = edgeStart + capsuloids.size();
    uint maxUniqueIdx = triangleStart + triangles.size();
  

    while (!outsideEverything)
    {
        // storing last position, because it may vary when it is pushed by multiple primitives
        glm::vec3 lastPos = lastPoint.pos;
        int tempDimensionality = -1;
        for (size_t uniqueIdx = 0; uniqueIdx < maxUniqueIdx; uniqueIdx++)
        {
            if (uniqueIdx >= singletonStart && uniqueIdx < edgeStart)
            {
                Point tempPoint = pushOutsideOneSingleton(uniqueIdx, lastPos, tempDimensionality);
                if (tempDimensionality != -1)
                {
                    // has been pushed outside
                    // break loop and restart it
                    lastDimensionality = tempDimensionality;
                    lastPoint = tempPoint;
                    break;
                }
            }
            else if (uniqueIdx >= edgeStart && uniqueIdx < triangleStart)
            {
                Point tempPoint = pushOutsideOneCapsule(uniqueIdx - edgeStart, lastPos, tempDimensionality);
                if (tempDimensionality != -1)
                {
                    // has been pushed outside
                    // break loop and restart it
                    lastDimensionality = tempDimensionality;
                    lastPoint = tempPoint;
                    break;
                }
            }
            
            /* Commented until pushOutsideOneTriangle is implemented
            else if (uniqueIdx >= triangleStart)
            {
                Point tempPoint = pushOutsideOneTriangle(uniqueIdx - triangleStart, lastPos, tempDimensionality);
                if (tempDimensionality != -1)
                {
                    // has been pushed outside
                    // break loop on and restart it
                    lastDimensionality = tempDimensionality;
                    lastPoint = tempPoint;
                    break;
                }
            }*/
        }

        outsideEverything = tempDimensionality == -1;
    }
    dimensionality = lastDimensionality;
    return lastPoint;
}
Point SphereMesh::pushOutsideOneCapsule(uint capsuleIndex, const glm::vec3 &pos, int &dimensionality) const
{
    const Capsuloid &caps = capsuloids.at(capsuleIndex);
    const Sphere &A = spheres.at(caps.s0);
    const Sphere &B = spheres.at(caps.s1);

    const glm::vec3 BminusA = B.center - A.center;
    const float BminusAsqrd = glm::dot(BminusA, BminusA);
    float k = glm::dot(pos - A.center, BminusA) / BminusAsqrd;

    // TODO va normalizzato BMinusA?
    const float factor = (A.radius - B.radius) / length(BminusA);

    k -= factor * length(pos - (A.center + k * BminusA));

    const float clampedK = glm::clamp(k, 0.0f, 1.0f);

    const glm::vec3 C = A.center + clampedK * BminusA;

    const glm::vec3 CtoPos = pos - C;

    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    const float interpRadius = A.radius * (1.0f - clampedK) + B.radius * clampedK;

    // pos is outside the capsule, dimensionality is -1 (not pushed out)
    // controllo con epsilon, se è sulla superficie non lo spingo
    if (CtoPossqrd > interpRadius * interpRadius - EPSILON)
    {
        dimensionality = -1;
        return Point(pos, glm::vec3(0.0f));
    }

    // if we are here, pos is inside the capsule
    // dimensionality depends on K value
    // if clampedK == k then pos is inside the cylinder, so dimensionality = 1
    // else pos is inside one of the spheres, so dimensionality = 0
    dimensionality = k == clampedK ? 1 : 0;
    const glm::vec3 normal = glm::normalize(CtoPos);

    return Point(glm::vec3(C + interpRadius * normal), normal);
}

Point SphereMesh::pushOutsideOneSingleton(uint singletonIndex, const glm::vec3 &pos, int &dimensionality) const
{
    const uint sphereIdx = singletons.at(singletonIndex);
    const Sphere &sphere = spheres.at(sphereIdx);

    const glm::vec3 &C = sphere.center;
    const glm::vec3 CtoPos = pos - C;
    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    // pos is outside sphere
    if (CtoPossqrd > sphere.radius * sphere.radius - EPSILON)
    {
        dimensionality = -1;
        return Point(pos, glm::vec3(0.0f));
    }

    // if we are here, pos is inside the sphere
    dimensionality = 0;
    const glm::vec3 normal = glm::normalize(CtoPos);
    return Point(glm::vec3(C + sphere.radius * normal), normal);
}

void SphereMesh::updateCapsuloidFactor(uint capsuloidIndex) {
    Capsuloid& c = capsuloids.at(capsuloidIndex);
    const Sphere& s0 = spheres.at(c.s0);
    const Sphere& s1 = spheres.at(c.s1);
    c.factor = computeCapsuloidFactor(c.s0, c.s1, glm::length(s0.center - s1.center));
}

void SphereMesh::updateAllCapsuloidsFactors() {
    for (Capsuloid& c : capsuloids) {
        const Sphere& s0 = spheres.at(c.s0);
        const Sphere& s1 = spheres.at(c.s1);
        c.factor = computeCapsuloidFactor(c.s0, c.s1, glm::length(s0.center - s1.center));
    }
}

bool readFromFile(const std::string &path, SphereMesh &out, std::string& errorMsg)
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

        uint trianglesNo = 0;
        file >> trianglesNo;
        for (size_t i = 0; i < trianglesNo; i++)
        {
            Triangle triangle;
            file >> triangle;
            out.addTriangle(triangle);
        }
    } catch (const std::ios::failure& ex) {
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
    ost << sm.triangles.size() << "\n";
    for (const auto &t : sm.triangles)
    {
        ost << t << "\n";
    }
    return ost;
}

float computeCapsuloidFactor(float r0, float r1, float dist) {
    return (r0 - r1) / dist;
}

