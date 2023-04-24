#pragma once

#include <vector>

#include <glm/vec3.hpp>

#include <spheremeshes/point.h>
#include <rendering/shader.h>

// vogliamo che PointCloud non sappia cos'è una sfera, ma solo cos'è un punto
// forward declaration nell'.h poi nel .cpp serve include però

class SphereMesh; // <-- forward declaration per cose secondarie

class PointCloud {
    private:
        std::vector<DimensionalityPoint> points;
        //invoca pushOutside della spheremesh, se ritorna dimensionality!=-1 allora da un colore altrimenti rirolla il punto
        void addPoint(const SphereMesh& sphereMesh);
    public:
        //itera per nPoints generate random inside della sfera passata
        // se nPoints > point.size(), aggiungo nPoints-puntiattuali punti
        // viceversa clampo
        void repopulate(const unsigned int nPoints, const SphereMesh& sphereMesh);
        void clear();
        unsigned int getPointsNumber() const;
        const void* pointerToData() const;
};
