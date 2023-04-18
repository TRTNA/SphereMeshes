#include <spheremeshes/spheremesh.h>

#include <iostream>
#include <stdio.h>
#include <sstream>
#include <array>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

using std::clog;
using std::endl;

using std::vector;
using std::array;
using std::ostream;
using std::stringstream;

static const float EPSILON = 0.001f;

SphereMesh::SphereMesh(vector<Sphere>& pSpheres, vector<Edge>& pEdges, vector<Triangle>& pTriangles, vector<uint>& pSingletons)
     : spheres(std::move(pSpheres)), edges(std::move(pEdges)), triangles(std::move(pTriangles)) , singletons(std::move(pSingletons))
{
    clog << "Created a sphere mesh:\n";
    clog << "- Spheres:\t" << spheres.size() << "\n";
    clog << "- Edges:\t" << edges.size() <<"\n";
    clog << "- Triangles:\t" << triangles.size() <<"\n";
    clog << "- Singletons:\t" << singletons.size() << "\n";
    updateBoundingSphere();
}

void SphereMesh::addSphere(const Sphere& sphere) {
    spheres.emplace_back(sphere.center, sphere.radius);
}

void SphereMesh::addEdge(const Edge& edge) {
    edges.emplace_back(edge.first, edge.second);
}

void SphereMesh::addTriangle(const Triangle& triangle) {
    triangles.emplace_back(triangle.vertices);
}

void SphereMesh::addSingleton(uint sphereIdx) {
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
    for(size_t i = 0; i < spheres.size(); i++) {
        ss << i << " " << spheres.at(i) << "\n";
    }
    ss << "\n";
    ss << "Edges:\n";
    for(size_t i = 0; i < edges.size(); i++) {
        ss << i << " " << edges.at(i) << "\n";
    }
    ss << "\n";
    ss << "Triangles:\n";
    for(size_t i = 0; i < triangles.size(); i++) {
        ss << i << " " << triangles.at(i) << "\n";
    }
    ss << "\n";
    ss << "Singletons:\n";
    for(size_t i = 0; i < singletons.size(); i++) {
        ss << i << " " << singletons.at(i) << "\n";
    }
    return ss.str();
}
Point SphereMesh::pushOutside(const glm::vec3 &pos, int &dimensionality) const
{
    //printf("\n\n[METHOD] pushOutside \n");
    //should iterate over all primitives and perform the right push out
    //for now it iterates only on capsules
    bool outsideEverything = false;
    Point lastPoint = Point(pos, glm::vec3(0.0f));
    int lastDimensionality = -1;
    while (! outsideEverything) {
        //storing last position, because it may vary when it is pushed by multiple primitives
        glm::vec3 lastPos = lastPoint.pos;
        //printf("[POINT LOOP] Params: - outsideEverything: %s\n - lastPoint: %s\n - lastDimensionality: %d\n", outsideEverything ? "true" : "false", glm::to_string(lastPos).c_str(), lastDimensionality);
        int tempDimensionality = -1;
        for (size_t idx = 0; idx < edges.size(); idx++) {
            Point tempPoint = pushOutsideOneCapsule(idx, lastPos, tempDimensionality);
            //printf("[EDGE LOOP] point pushed by %d edge, new value %s and dimensionality %d\n", idx, glm::to_string(tempPoint.pos).c_str(), tempDimensionality);
            if (tempDimensionality != -1) {
                //has been pushed outside
                //break loop on edges and restart it
                //printf("[PUSHED] point has been pushed outside\n");
                lastDimensionality = tempDimensionality;
                lastPoint = tempPoint;
                break;
            }
        }
        for (size_t idx = 0; idx < singletons.size(); idx++) {
            Point tempPoint = pushOutsideOneSingleton(idx, lastPos, tempDimensionality);
            //printf("[EDGE LOOP] point pushed by %d edge, new value %s and dimensionality %d\n", idx, glm::to_string(tempPoint.pos).c_str(), tempDimensionality);
            if (tempDimensionality != -1) {
                //has been pushed outside
                //break loop on edges and restart it
                //printf("[PUSHED] point has been pushed outside\n");
                lastDimensionality = tempDimensionality;
                lastPoint = tempPoint;
                break;
            }
        }
        outsideEverything = tempDimensionality == -1;

    }
    dimensionality = lastDimensionality;
    return lastPoint;

}
Point SphereMesh::pushOutsideOneCapsule(uint capsuleIndex, const glm::vec3 &pos, int &dimensionality) const
{
    const Edge& edge = edges.at(capsuleIndex);
    const Sphere& A = spheres.at(edge.first);
    const Sphere& B = spheres.at(edge.second);

    const glm::vec3 BminusA = B.center - A.center;
    const float BminusAsqrd = glm::dot(BminusA, BminusA);
    float k = glm::dot(pos - A.center, BminusA) / BminusAsqrd;

    //TODO va normalizzato BMinusA?
    const float factor = (A.radius - B.radius) / length(BminusA);

    k -= factor * length(pos - (A.center + k*BminusA));

    const float clampedK = glm::clamp(k, 0.0f, 1.0f);

    const glm::vec3 C = A.center + clampedK*BminusA;

    const glm::vec3 CtoPos = pos - C;

    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    const float interpRadius = A.radius * (1.0f - clampedK) + B.radius * clampedK;

    //pos is outside the capsule, dimensionality is -1 (not pushed out)
    //controllo con epsilon, se è sulla superficie non lo spingo
    if (CtoPossqrd > interpRadius*interpRadius - EPSILON) {
        dimensionality = -1;
        return Point(pos, glm::vec3(0.0f));
    }

    //if we are here, pos is inside the capsule
    //dimensionality depends on K value
    //if clampedK == k then pos is inside the cylinder, so dimensionality = 1
    //else pos is inside one of the spheres, so dimensionality = 0
    dimensionality = k == clampedK ? 1 : 0;
    const glm::vec3 normal = glm::normalize(CtoPos);

    return Point(glm::vec3(C + interpRadius*normal), normal);
}

Point SphereMesh::pushOutsideOneSingleton(uint singletonIndex, const glm::vec3& pos, int& dimensionality) const {
    const uint sphereIdx = singletons.at(singletonIndex);
    const Sphere& sphere = spheres.at(sphereIdx);

    const glm::vec3& C = sphere.center;
    const glm::vec3 CtoPos = pos - C;
    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    //pos is outside sphere
    if (CtoPossqrd > sphere.radius*sphere.radius - EPSILON) {
        dimensionality = -1;
        return Point(pos, glm::vec3(0.0f));
    }

    //if we are here, pos is inside the sphere
    dimensionality = 0;
    const glm::vec3 normal = glm::normalize(CtoPos);
    return Point(glm::vec3(C + sphere.radius*normal), normal);
}


std::ostream& operator<<(std::ostream& ost, const SphereMesh& sm) {
    ost << sm.toString();
    return ost;
}


