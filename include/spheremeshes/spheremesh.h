#pragma once

#include <vector>
#include <string>
#include <ostream>

#include <glm/glm.hpp>


#include <spheremeshes/sphere.h>
#include <spheremeshes/capsuloid.h>
#include <spheremeshes/spheretriangle.h>
#include <spheremeshes/point.h>

typedef unsigned int uint;

static const float EPSILON = 0.00001f;


class SphereMesh {
    public:
    std::vector<Sphere> spheres;
    std::vector<uint> singletons;
    std::vector<Capsuloid> capsuloids;
    std::vector<SphereTriangle> sphereTriangles;
    void scale(float k);
    Sphere boundingSphere;
    SphereMesh() = default;
    SphereMesh(std::vector<Sphere>& pSpheres, std::vector<Capsuloid>& pEdges, std::vector<SphereTriangle>& pTriangles, std::vector<uint>& pSingletons);
    ~SphereMesh() = default;
    void addSphere(Sphere sphere);
    void addCapsuloid(Capsuloid capsuloid);
    void addSphereTriangle(SphereTriangle sphereTriangle);
    void addSingleton(uint sphereIdx);
    void updateBoundingSphere();
    std::string toString() const;
    //serve sapere la dimensionalità di cosa l'ha spinto fuori
    // dimensionality = 0 = sfera, = 1 =segmento, =2=triangolo, =-1=ero già fuori
    // itero push fuori da tutto e mi fermo quando?
    // getRandomPointOnSurface --> genera un punto sulla superficie della sfera colorato a seconda di chi lo ha spinto fuori
    // genera un punto, prova a spingerlo fuori, se nessuno lo spinge fuori riprova, altrimenti assegna colore e ritorna.
    Point pushOutside(const glm::vec3& pos, int& dimensionality) const;
    private:
    Point pushOutsideOneCapsule(const Capsuloid& caps, const glm::vec3& pos, int& dimensionality) const;
    Point pushOutsideOneSphereTriangle(const SphereTriangle& tri, const glm::vec3& pos, int& dimensionality) const;
    Point pushOutsideOneSingleton(const Sphere& sphere, const glm::vec3& pos, int& dimensionality) const;
    void updateAllCapsuloidsFeatures();
    void updateAllSphereTriangleFeatures();

};


void updateCapsuloidFeatures(Capsuloid& caps, const Sphere& s0, const Sphere& s1);
void updateSphereTriangleFeatures(SphereTriangle& tri, const Sphere& s0, const Sphere& s1, const Sphere& s2);

//Transform a position (passed as q which is the vector from the center of the sphere0 to the position) 
// in the reference system of the sphere triangle, computing the barycentric coordinates (outA, outB and outC) and the distance from the plane
void toSphereTriangleReferenceSystem(const SphereTriangle& tri, const glm::vec3& q, float& outA, float& outB, float& outC, float& outD);
Point pointOutsideSphereMesh(const glm::vec3& pos, int &dimensionality);


// I/O
bool readFromFile(const std::string& path, SphereMesh& out, std::string& errorMsg);
std::ostream& operator<<(std::ostream& ost, const SphereMesh& sm);
std::istream& operator>>(std::istream& ost, SphereMesh& sm);
