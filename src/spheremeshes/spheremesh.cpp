#include <spheremeshes/spheremesh.h>
#include <utils/common.h>

#include <iostream>
#include <stdio.h>
#include <sstream>

#include <array>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>

using std::clog;
using std::endl;

using glm::vec3;
using std::array;
using std::ostream;
using std::string;
using std::stringstream;
using std::vector;



SphereMesh::SphereMesh(vector<Sphere> &pSpheres, vector<Capsuloid> &pCapsuloids, vector<SphereTriangle> &pSphereTriangles, vector<uint> &pSingletons)
    : spheres(std::move(pSpheres)), capsuloids(std::move(pCapsuloids)), sphereTriangles(std::move(pSphereTriangles)), singletons(std::move(pSingletons))
{
    clog << "Created a sphere mesh:\n";
    clog << "- Spheres:\t" << spheres.size() << "\n";
    clog << "- Capsuloids:\t" << capsuloids.size() << "\n";
    clog << "- Triangles:\t" << sphereTriangles.size() << "\n";
    clog << "- Singletons:\t" << singletons.size() << "\n";
    updateAllCapsuloidsFeatures();
    updateAllSphereTriangleFeatures();
    updateBoundingSphere();
}

void SphereMesh::addSphere(Sphere sphere)
{
    spheres.emplace_back(sphere.center, sphere.radius);
}

void SphereMesh::addCapsuloid(Capsuloid caps)
{   
    updateCapsuloidFeatures(caps, spheres.at(caps.s0), spheres.at(caps.s1));
    capsuloids.push_back(caps);
}

void SphereMesh::addSphereTriangle(SphereTriangle st)
{
    updateSphereTriangleFeatures(st, spheres.at(st.s0), spheres.at(st.s1), spheres.at(st.s2));
    sphereTriangles.push_back(st);
}

void SphereMesh::addSingleton(uint sphereIdx)
{
    singletons.push_back(sphereIdx);
}

void SphereMesh::updateBoundingSphere()
{
    boundingSphere = computeBoundingSphere(spheres);
}

void SphereMesh::scale(float k)
{
    for (auto &s : spheres)
    {
        s.scale(k);
    }
    boundingSphere.scale(k);
    updateAllCapsuloidsFeatures();
    updateAllSphereTriangleFeatures();
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
    while (!outsideEverything && tries < maxPushOutsideTries)
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
    if (tries == maxPushOutsideTries) {
        dimensionality = -1;
        return lastPoint;
    }
    dimensionality = lastDimensionality;
    return lastPoint;
}

Point SphereMesh::pushOutsideOneCapsule(const Capsuloid &caps, const vec3 &pos, int &dimensionality) const
{
    const Sphere &A = spheres.at(caps.s0);
    const Sphere &B = spheres.at(caps.s1);

    const vec3& BminusA = caps.S0toS1;
    float k = glm::dot(pos - A.center, BminusA) / caps.sqrdL;
    vec3 fakeC = A.center + k * BminusA;
    float d = length(fakeC - pos);

    k += (caps.factor * d);

    const float clampedK = glm::clamp(k, 0.0f, 1.0f);

    const vec3 C = A.center + clampedK * BminusA;

    const vec3 CtoPos = pos - C;

    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    const float interpRadius = A.radius * (1.0f - clampedK) + B.radius * clampedK;

    // pos is outside the capsule, dimensionality is -1 (not pushed out)
    if (CtoPossqrd > interpRadius * interpRadius - EPSILON)
    {
        return pointOutsideSphereMesh(pos, dimensionality);
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
        return pointOutsideSphereMesh(pos, dimensionality);
    }

    // if we are here, pos is inside the sphere
    dimensionality = 0;
    const vec3 normal = glm::normalize(CtoPos);
    return Point(vec3(C + sphere.radius * normal), normal);
}

Point SphereMesh::pushOutsideOneSphereTriangle(const SphereTriangle &tri, const vec3 &pos, int &dimensionality) const
{
    const Sphere &s0 = spheres.at(tri.s0);
    const Sphere &s1 = spheres.at(tri.s1);
    const Sphere &s2 = spheres.at(tri.s2);


    const vec3 q = pos - s0.center;
    float d, a, b, c;
    toSphereTriangleReferenceSystem(tri, q, a, b, c, d);
    
    if (b < 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V0V1
        Capsuloid &tempCapsule = Capsuloid(tri.s0, tri.s1);
        updateCapsuloidFeatures(tempCapsule, s0, s1);
        return pushOutsideOneCapsule(tempCapsule, pos, dimensionality);
    }
    if (c < 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V1V2
        Capsuloid &tempCapsule = Capsuloid(tri.s1, tri.s2);
        updateCapsuloidFeatures(tempCapsule, s1, s2);
        return pushOutsideOneCapsule(tempCapsule, pos, dimensionality);
    }
    if (a < 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V0V2
        Capsuloid &tempCapsule = Capsuloid(tri.s0, tri.s2);
        updateCapsuloidFeatures(tempCapsule, s0, s2);
        return pushOutsideOneCapsule(tempCapsule, pos, dimensionality);
    }
    //FIXME: già controllato e nessuno sarà minore di zero, si potrebbe controllare solo che siano minori di 1.0 incluso e risparmiare tre confronti
    //PUSH OUTSIDE TRIANGLE
    if(isInRangeIncl(a, 0.0f, 1.0f) && isInRangeIncl(b, 0.0f, 1.0f) && isInRangeIncl(c, 0.0f, 1.0f))
    {
        vec3 C = c * s0.center + a * s1.center + b * s2.center;
        float interpRadius = c * s0.radius + a * s1.radius + b * s2.radius;
        vec3 CtoPos = pos - C;
        //float distPosC = glm::length(CtoPos);
        if (d > interpRadius - EPSILON)
        {
            return pointOutsideSphereMesh(pos, dimensionality);
        }
        dimensionality = 2;
        glm::vec3 normal = glm::normalize(CtoPos);
        return Point(C + interpRadius * normal, normal);
    }

    return pointOutsideSphereMesh(pos, dimensionality);
}

void SphereMesh::setMaxPushOutsideTries(uint val) {
    maxPushOutsideTries = val;
}



void SphereMesh::updateAllCapsuloidsFeatures()
{
    for (Capsuloid &c : capsuloids)
    {
        const Sphere &s0 = spheres.at(c.s0);
        const Sphere &s1 = spheres.at(c.s1);
        updateCapsuloidFeatures(c, s0, s1);
    }
}

void SphereMesh::updateAllSphereTriangleFeatures()
{
    for (SphereTriangle &st : sphereTriangles)
    {
        updateSphereTriangleFeatures(st, spheres.at(st.s0), spheres.at(st.s1), spheres.at(st.s2));
    }
}

