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

using glm::vec3;
using std::array;
using std::ifstream;
using std::ostream;
using std::string;
using std::stringstream;
using std::vector;

static const float EPSILON = 0.00001f;

SphereMesh::SphereMesh(vector<Sphere> &pSpheres, vector<Capsuloid> &pCapsuloids, vector<SphereTriangle> &pSphereTriangles, vector<uint> &pSingletons)
    : spheres(std::move(pSpheres)), capsuloids(std::move(pCapsuloids)), sphereTriangles(std::move(pSphereTriangles)), singletons(std::move(pSingletons))
{
    clog << "Created a sphere mesh:\n";
    clog << "- Spheres:\t" << spheres.size() << "\n";
    clog << "- Capsuloids:\t" << capsuloids.size() << "\n";
    clog << "- Triangles:\t" << sphereTriangles.size() << "\n";
    clog << "- Singletons:\t" << singletons.size() << "\n";
    updateAllCapsuloidsFactors();
    updateAllSphereTrianglesProjMat();
    updateBoundingSphere();
}

void SphereMesh::addSphere(const Sphere &sphere)
{
    spheres.emplace_back(sphere.center, sphere.radius);
}

void SphereMesh::addCapsuloid(const Capsuloid &caps)
{
    capsuloids.emplace_back(caps.s0, caps.s1, computeCapsuloidFactor(spheres.at(caps.s0), spheres.at(caps.s1)));
}

void SphereMesh::addSphereTriangle(const SphereTriangle &st)
{
    const Sphere &s0 = spheres.at(st.vertices[0]);
    const Sphere &s1 = spheres.at(st.vertices[1]);
    const Sphere &s2 = spheres.at(st.vertices[2]);
    sphereTriangles.emplace_back(st.vertices, computeSphereTriangleProjMat(s0.center, s1.center, s2.center));
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
    ss << "SphereTriangles:\n";
    for (size_t i = 0; i < sphereTriangles.size(); i++)
    {
        ss << i << " " << sphereTriangles.at(i) << "\n";
    }
    ss << "\n";
    ss << "Singletons:\n";
    for (size_t i = 0; i < singletons.size(); i++)
    {
        ss << i << " " << singletons.at(i) << "\n";
    }
    return ss.str();
}
Point SphereMesh::pushOutside(const vec3 &pos, int &dimensionality) const
{
    bool outsideEverything = false;
    Point lastPoint = Point(pos, vec3(0.0f));
    int lastDimensionality = -1;
    uint singletonStart = 0;
    uint edgeStart = singletonStart + singletons.size();
    uint triangleStart = edgeStart + capsuloids.size();
    uint maxUniqueIdx = triangleStart + sphereTriangles.size();
    uint tries = 0;
    const uint maxTries = 10;

    while (!outsideEverything && tries < maxTries)
    {
        // storing last position, because it may vary when it is pushed by multiple primitives
        vec3 lastPos = lastPoint.pos;
        int tempDimensionality = -1;
        for (size_t uniqueIdx = 0; uniqueIdx < maxUniqueIdx; uniqueIdx++)
        {
            if (uniqueIdx >= singletonStart && uniqueIdx < edgeStart)
            {
                Point tempPoint = pushOutsideOneSingleton(spheres.at(singletons.at(uniqueIdx)), lastPos, tempDimensionality);
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
                Point tempPoint = pushOutsideOneCapsule(capsuloids.at(uniqueIdx - edgeStart), lastPos, tempDimensionality);
                if (tempDimensionality != -1)
                {
                    // has been pushed outside
                    // break loop and restart it
                    lastDimensionality = tempDimensionality;
                    lastPoint = tempPoint;
                    break;
                }
            }

            // Commented until pushOutsideOneSphereTriangle is implemented
            else if (uniqueIdx >= triangleStart)
            {
                Point tempPoint = pushOutsideOneSphereTriangle(sphereTriangles.at(uniqueIdx - triangleStart), lastPos, tempDimensionality);
                if (tempDimensionality != -1)
                {
                    // has been pushed outside
                    // break loop on and restart it
                    lastDimensionality = tempDimensionality;
                    lastPoint = tempPoint;
                    break;
                }
            }
        }
        tries++;
        outsideEverything = tempDimensionality == -1;
    }
    dimensionality = lastDimensionality;
    return lastPoint;
}

Point SphereMesh::pushOutsideOneCapsule(const Capsuloid &caps, const vec3 &pos, int &dimensionality) const
{
    const Sphere &A = spheres.at(caps.s0);
    const Sphere &B = spheres.at(caps.s1);

    const vec3 BminusA = B.center - A.center;
    const float BminusAsqrd = glm::dot(BminusA, BminusA);
    float k = glm::dot(pos - A.center, BminusA) / BminusAsqrd;
    vec3 fakeC = A.center + k * BminusA;
    float d = length(fakeC - pos);

    k += (caps.factor * d);

    const float clampedK = glm::clamp(k, 0.0f, 1.0f);

    const vec3 C = A.center + clampedK * BminusA;

    const vec3 CtoPos = pos - C;

    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    const float interpRadius = A.radius * (1.0f - clampedK) + B.radius * clampedK;

    // pos is outside the capsule, dimensionality is -1 (not pushed out)
    // controllo con epsilon, se è sulla superficie non lo spingo
    if (CtoPossqrd > interpRadius * interpRadius - EPSILON)
    {
        dimensionality = -1;
        return Point(pos, vec3(0.0f));
    }

    // if we are here, pos is inside the capsule
    // dimensionality depends on K value
    // if clampedK == k then pos is inside the cylinder, so dimensionality = 1
    // else pos is inside one of the spheres, so dimensionality = 0
    dimensionality = k == clampedK ? 1 : 0;
    const vec3 normal = glm::normalize(CtoPos);

    return Point(vec3(C + interpRadius * normal), normal);
}

Point SphereMesh::pushOutsideOneSingleton(const Sphere &sphere, const vec3 &pos, int &dimensionality) const
{
    const vec3 &C = sphere.center;
    const vec3 CtoPos = pos - C;
    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    // pos is outside sphere
    if (CtoPossqrd > sphere.radius * sphere.radius - EPSILON)
    {
        dimensionality = -1;
        return Point(pos, vec3(0.0f));
    }

    // if we are here, pos is inside the sphere
    dimensionality = 0;
    const vec3 normal = glm::normalize(CtoPos);
    return Point(vec3(C + sphere.radius * normal), normal);
}

Point SphereMesh::pushOutsideOneSphereTriangle(const SphereTriangle &tri, const vec3 &pos, int &dimensionality) const
{
    const Sphere &s0 = spheres.at(tri.vertices[0]);
    const Sphere &s1 = spheres.at(tri.vertices[1]);
    const Sphere &s2 = spheres.at(tri.vertices[2]);

    const vec3 q = pos - s0.center;

    vec3 res = tri.projectorMatrix * q;
    float d = res.z;
    float k0 = res.x;
    float k1 = res.y;

    float a = k0;
    float b = k1;
    float c = (1.0f - k0 - k1);
    /*
        if (a <= 0.0f && b <= 0.0f)
        {
            // PUSH OUTSIDE SPHERE S0
            return pushOutsideOneSingleton(s0, pos, dimensionality);
        }

        if (b <= 0.0f && c <= 0.0f)
        {
            // PUSH OUTSIDE SPHERE S1
            return pushOutsideOneSingleton(s1, pos, dimensionality);
        }

        if (a <= 0.0f && c <= 0.0f)
        {
            // PUSH OUTSIDE SPHERE S2
            return pushOutsideOneSingleton(s2, pos, dimensionality);
        }

    */
    if (b <= 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V0V1
        Capsuloid &tempCapsule = Capsuloid(tri.vertices[0], tri.vertices[1], computeCapsuloidFactor(s0, s1));
        return pushOutsideOneCapsule(tempCapsule, pos, dimensionality);
    }
    if (c <= 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V1V2
        Capsuloid &tempCapsule = Capsuloid(tri.vertices[1], tri.vertices[2], computeCapsuloidFactor(s1, s2));
        return pushOutsideOneCapsule(tempCapsule, pos, dimensionality);
    }
    if (a <= 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V0V2
        Capsuloid &tempCapsule = Capsuloid(tri.vertices[0], tri.vertices[2], computeCapsuloidFactor(s0, s2));
        return pushOutsideOneCapsule(tempCapsule, pos, dimensionality);
    }

    if (a > 0.0f && a < 1.0f && b > 0.0f && b < 1.0f && c > 0.0f && c < 1.0f)
    {
        vec3 wrongProjection = c * s0.center + a * s1.center + b * s2.center;
        vec3 C0minusC1 = s0.center - s1.center;
        vec3 C2minusC1 = s2.center - s1.center;
        vec3 n = glm::normalize(glm::cross(-C0minusC1, C2minusC1));
        vec3 e = vec3(1.0f, 1.0f, 1.0f);
        vec3 oldN = n;
        do
        {
            glm::mat3 A = glm::mat3(C0minusC1, C2minusC1, n);
            A = glm::transpose(A); // put rows as columns because glm is row major, so transposition is needed
            vec3 t = vec3(
                s1.radius - s0.radius - glm::dot(C0minusC1, n),
                s1.radius - s2.radius - glm::dot(C2minusC1, n),
                0.0f);
            e = t * glm::inverse(A);
            n = glm::normalize(n + e);
        } while (e.x > EPSILON && e.y > EPSILON && e.z > EPSILON);

        float cosAlpha = glm::dot(n, oldN);
        float projDist = d / cosAlpha;

        vec3 C = pos + projDist * (-n);

        // for now exclude point under the plane
        if (glm::dot(pos - wrongProjection, oldN) < 0.0)
        {
            dimensionality = -1;
            return Point(pos, vec3(0.0f));
        }
        // PUSH OUTSIDE TRIANGLE
        float wrongInterpRadius = c * s0.radius + a * s1.radius + b * s2.radius;
        float correctInterpRadius = glm::sqrt(glm::pow(glm::length(wrongProjection - C), 2) + glm::pow(wrongInterpRadius, 2));
        if (projDist > correctInterpRadius - EPSILON)
        {
            dimensionality = -1;
            return Point(pos, vec3(0.0f));
        }
        dimensionality = 2;
        return Point(C + correctInterpRadius * n, n);
    }
    dimensionality = -1;
    return Point(pos, vec3(0.0f));
}

void SphereMesh::updateCapsuloidFactor(uint capsuloidIndex)
{
    Capsuloid &c = capsuloids.at(capsuloidIndex);
    const Sphere &s0 = spheres.at(c.s0);
    const Sphere &s1 = spheres.at(c.s1);
    c.factor = computeCapsuloidFactor(s0, s1);
}

void SphereMesh::updateAllCapsuloidsFactors()
{
    for (Capsuloid &c : capsuloids)
    {
        const Sphere &s0 = spheres.at(c.s0);
        const Sphere &s1 = spheres.at(c.s1);
        c.factor = computeCapsuloidFactor(s0, s1);
    }
}

void SphereMesh::updateSphereTriangleProjMat(uint sphereTriangleIndex)
{
    SphereTriangle &st = sphereTriangles.at(sphereTriangleIndex);
    const Sphere &s0 = spheres.at(st.vertices[0]);
    const Sphere &s1 = spheres.at(st.vertices[1]);
    const Sphere &s2 = spheres.at(st.vertices[2]);

    st.setProjectorMatrix(computeSphereTriangleProjMat(s0.center, s1.center, s2.center));
}
void SphereMesh::updateAllSphereTrianglesProjMat()
{
    for (SphereTriangle &st : sphereTriangles)
    {
        const Sphere &s0 = spheres.at(st.vertices[0]);
        const Sphere &s1 = spheres.at(st.vertices[1]);
        const Sphere &s2 = spheres.at(st.vertices[2]);
        st.setProjectorMatrix(computeSphereTriangleProjMat(s0.center, s1.center, s2.center));
    }
}

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

glm::mat3 computeSphereTriangleProjMat(const vec3 &v0, const vec3 v1, const vec3 &v2)
{
    const vec3 A = v1 - v0;
    const vec3 B = v2 - v0;
    const vec3 N = glm::normalize(glm::cross(A, B));
    return glm::inverse(glm::mat3(A, B, N));
}

void SphereMesh::scale(float k)
{
    for (auto &s : spheres)
    {
        s.scale(k);
    }
    boundingSphere.scale(k);
    updateAllCapsuloidsFactors();
    updateAllSphereTrianglesProjMat();
}
