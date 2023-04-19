#pragma once

#include <vector>
#include <string>
#include <ostream>

#include <spheremeshes/sphere.h>
#include <spheremeshes/capsuloid.h>
#include <spheremeshes/triangle.h>


#include <spheremeshes/point.h>

typedef unsigned int uint;

class SphereMesh {
    public:
    std::vector<Sphere> spheres;
    std::vector<uint> singletons;
    std::vector<Capsuloid> capsuloids;
    std::vector<Triangle> triangles;
    //TODO moltiplica tutti i punti e raggi per k (voledno metodo scale su sphere da invocare su ognuna)
    //e aggiorna boundingsphere subito dopo, scala anche la boundingSphere
    void scale(float k);
    Sphere boundingSphere;
    SphereMesh() = default;
    SphereMesh(std::vector<Sphere>& pSpheres, std::vector<Capsuloid>& pEdges, std::vector<Triangle>& pTriangles, std::vector<uint>& pSingletons);
    ~SphereMesh() = default;
    void addSphere(const Sphere& sphere);
    void addCapsuloid(const Capsuloid& capsuloid);
    void addTriangle(const Triangle& triangle);
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
    Point pushOutsideOneCapsule(uint capsuloidIndex, const glm::vec3& pos, int& dimensionality) const;
    Point pushOutsideOneTriangle(uint triangleIndex, const glm::vec3& pos, int& dimensionality) const;
    Point pushOutsideOneSingleton(uint singletonIndex, const glm::vec3& pos, int& dimensionality) const;
    void updateCapsuloidFactor(uint capsuloidIndex);
    void updateAllCapsuloidsFactors();
};


bool readFromFile(const std::string& path, SphereMesh& out, std::string& errorMsg);
float computeCapsuloidFactor(float r0, float r1, float dist);



std::ostream& operator<<(std::ostream& ost, const SphereMesh& sm);
std::istream& operator>>(std::istream& ost, SphereMesh& sm);
