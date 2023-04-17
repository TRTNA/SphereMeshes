#pragma once

#include <vector>
#include <string>
#include <ostream>



#include <spheremeshes/sphere.h>
#include <spheremeshes/edge.h>
#include <spheremeshes/triangle.h>

#include <utils/model.h>
#include <utils/shader.h>

#include <spheremeshes/point.h>

typedef unsigned int uint;

class SphereMesh {
    public:
    std::vector<Sphere> spheres;
    std::vector<uint> singleton;
    std::vector<Edge> edges;
    std::vector<Triangle> triangles;
    //TODO moltiplica tutti i punti e raggi per k (voledno metodo scale su sphere da invocare su ognuna)
    //e aggiorna boundingsphere subito dopo, scala anche la boundingSphere
    void scale(float k);
    Sphere boundingSphere;
    SphereMesh() = default;
    SphereMesh(std::vector<Sphere>& pSpheres, std::vector<Edge>& pEdges, std::vector<Triangle>& pTriangles);
    ~SphereMesh() = default;
    void addSphere(const Sphere& phere);
    void addEdge(const Edge& edge);
    void addTriangle(const Triangle& triangle);
    void updateBoundingSphere();
    std::string toString() const;
    //serve sapere la dimensionalità di cosa l'ha spinto fuori
    // dimensionality = 0 = sfera, = 1 =segmento, =2=triangolo, =-1=ero già fuori
    // itero push fuori da tutto e mi fermo quando?
    // getRandomPointOnSurface --> genera un punto sulla superficie della sfera colorato a seconda di chi lo ha spinto fuori
    // genera un punto, prova a spingerlo fuori, se nessuno lo spinge fuori riprova, altrimenti assegna colore e ritorna.
    Point pushOutside(const glm::vec3& pos, int& dimensionality) const;
    private:
    Point pushOutsideOneCapsule(uint capsuleIndex, const glm::vec3& pos, int& dimensionality) const;
    Point pushOutsideOneTriangle(uint triangleIndex, const glm::vec3& pos, int& dimensionality) const;
    Point pushOutsideOneSingleton(uint singletonIndex, const glm::vec3& pos, int& dimensionality) const;

};




std::ostream& operator<<(std::ostream& ost, const SphereMesh& sm);
