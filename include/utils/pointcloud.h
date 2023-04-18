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
        std::vector<ColoredPoint> points;
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


//TODO
// Metodo che costruisce point cloud popolandola con una sphere mesh
// prende bounding sphere della sphere mesh, la popola con punti casuali nella bounding sphere
// ogni punto lo butta fuori da tutti i componenti della sphere mesh (SU CPU) finchè quel punto non è sulla superficie di uno è fuori da tutti gli altri
// processo iterativo che continua a spingerlo finchè non è sulla superficie 
// metodi push outside sphere, capsule e triangle
