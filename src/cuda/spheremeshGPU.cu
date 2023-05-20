#include <cuda/spheremeshGPU.h>
#include <spheremeshes/point.h>
#include <spheremeshes/sphere.h>

#include <utils/random.h>

#include <glm/glm.hpp>

#include <stdio.h>

#include <cmath>

#include <array>

using std::array;

typedef unsigned long ulong;

float computeTime(cudaEvent_t &e1, cudaEvent_t &e2)
{
    float time;
    cudaEventElapsedTime(&time, e1, e2);
    return time;
}

void checkError(cudaError error)
{

    if (error != 0)
    {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error,
                cudaGetErrorString(error));
    }
}

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
        }                                                          \
    }

__device__ float dot(float x1, float y1, float z1, float x2, float y2, float z2)
{
    return x1 * x2 + y1 * y2 + z1 * z2;
}

__global__ void pushOutsideSphere(glm::vec3 *positions, glm::vec3 *normals, int *dimensionality, Sphere sphere)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    glm::vec3 pos = positions[tid];
    glm::vec3 CtoPos = pos - sphere.center;
    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    // pos is outside sphere
    if (CtoPossqrd > sphere.radius * sphere.radius - 0.0001f)
    {
        dimensionality[tid] = -2;
    }

    // if we are here, pos is inside the sphere
    dimensionality[tid] = -1;
    CtoPos = glm::normalize(CtoPos);
    positions[tid] = sphere.center + sphere.radius * CtoPos;
    normals[tid] = CtoPos;
}

__global__ void pushOutsideCapsuloid(float *dX, float *dY, float *dZ) {}

void createSphereMeshGPU(SphereMesh &sphereMesh, uint numberOfPoints, std::vector<DimensionalityPoint> &outPoints)
{
    printf("Starting...\n");
    cudaSetDevice(0);
    array<cudaEvent_t, 11> events;

    for (cudaEvent_t &event : events)
    {
        CHECK(cudaEventCreate(&event));
    }

    // # 1. Inizializzazione memoria host
    CHECK(cudaEventRecord(events[0], 0));
    printf("Inizializzazione memoria host...\n");
    ulong coordinatesBytes = numberOfPoints * sizeof(float) * 3;
    ulong dimensionalityBytes = numberOfPoints * sizeof(int);

    glm::vec3 *hostPositions = (glm::vec3 *)malloc(coordinatesBytes);
    glm::vec3 *hostNormals = (glm::vec3 *)malloc(coordinatesBytes);


    int *hostDimensionality = (int *)malloc(dimensionalityBytes);
    for (size_t i = 0; i < numberOfPoints; i++)
    {
    }
    CHECK(cudaEventRecord(events[1], 0));
    // wait until the stop event completes
    CHECK(cudaEventSynchronize(events[1]));
    printf("Allocati %lu bytes in memoria host in %f millisecondi...\n", coordinatesBytes + dimensionalityBytes, computeTime(events[0], events[1]));

    // # 2. TODO inizializzare i punti con valori random dentro la bounding sphere della sphere mesh
    // # on GPU?
    for (size_t i = 0; i < numberOfPoints; i++)
    {
        hostPositions[i].x = generateFloat(-2.0f, 2.0f);
        hostPositions[i].y = generateFloat(-2.0f, 2.0f);
        hostPositions[i].z = generateFloat(-2.0f, 2.0f);
        hostDimensionality[i] = -1;
    }

    // # 3. Inizializzazione memoria device
    printf("Inizializzazione memoria device...\n");
    int *deviceDimensionality;
    glm::vec3 *devicePositions, *deviceNormals;
    CHECK(cudaEventRecord(events[2]));

    CHECK(cudaMalloc((void **)&devicePositions, coordinatesBytes));
        CHECK(cudaMalloc((void **)&deviceNormals, coordinatesBytes));

    CHECK(cudaMalloc((void **)&deviceDimensionality, dimensionalityBytes));

    CHECK(cudaEventRecord(events[3]));
    CHECK(cudaEventSynchronize(events[3]));

    printf("Allocati %lu bytes in memoria device in %f millisecondi...\n", coordinatesBytes + dimensionalityBytes, computeTime(events[2], events[3]));

    // # 4. Copia da memoria host a memoria device
    printf("Copia dati da host a device...\n");
    CHECK(cudaEventRecord(events[4]));

    CHECK(cudaMemcpy(devicePositions, hostPositions, coordinatesBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceNormals, hostNormals, coordinatesBytes, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(deviceDimensionality, hostDimensionality, dimensionalityBytes, cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(events[5]));
    CHECK(cudaEventSynchronize(events[5]));

    printf("Copia dati da host a device TERMINATA in %f millisecondi\n", computeTime(events[4], events[5]));

    // # 6. Creazione contesto loop
    const uint maxTries = 10U;
    uint tries = 0U;
    bool *allNegativeDim;
    CHECK(cudaMallocManaged(&allNegativeDim, sizeof(int)));

    CHECK(cudaEventRecord(events[6]));

    int blockSize = 1024;
    dim3 grid((numberOfPoints / blockSize) + 1);
    dim3 block(blockSize);

    Sphere testSphere{glm::vec3(0.0f), 2.0f};

    // # 7. Loop di creazione dei punti
    do
    {
        printf("Tentativo %u...", tries);
        // # 7.1 TODO Chiamata al kernel di push outside
        // # pushOutside<<<grid, block>>>(devicePoints, numberOfPoints);

        // # 7.2 Sincronizzazione sul lavoro del kernel pushOutside
        printf("Attesa terminazione kernel...\n");

        pushOutsideSphere<<<grid, block>>>(devicePositions, deviceNormals, deviceDimensionality, testSphere);
        checkError(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // # 7.3 TODO Chiamata al kernel che controlla se sono tutte negative le dimensionality
        printf("Controllo dimensionalita' punti...\n");
        // # checkAllNegativeDimensionalities<<<grid, block>>>(devicePoints, numberOfPoints, allNegativeDim);
        tries++;
    } while (!(*allNegativeDim) && tries < maxTries);
    // # Esce dal ciclo quanto tutti i punti sono o esterni alla sphere mesh o spinti sulla superficie
    CHECK(cudaEventRecord(events[7]));
    CHECK(cudaEventSynchronize(events[7]));

    printf("Creazione punti TERMINATA in %f millisecondi\n", computeTime(events[6], events[7]));
    printf("Creazione punti terminata perche' %s\n", tries == maxTries ? "sono stati esauriti i tentativi" : "i punti sono tutti esterni o sulla superficie");

    // # 8. Copia da memoria device a memoria host (funziona così?)
    printf("Copia dati da device a host...\n");
    CHECK(cudaEventRecord(events[8]));

    CHECK(cudaMemcpy(hostPositions, devicePositions, coordinatesBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostNormals, deviceNormals, coordinatesBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(hostDimensionality, deviceDimensionality, dimensionalityBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(events[9]));
    CHECK(cudaEventSynchronize(events[9]));

    printf("Copia dati da device a host TERMINATA in %f millisecondi\n", computeTime(events[8], events[9]));

    // # 9. Eliminazione memoria allocata sul device (memcpy è bloccante, sono sicuro che non mi serva più quando arrivo qui)

    CHECK(cudaFree(devicePositions));
        CHECK(cudaFree(deviceNormals));

    CHECK(cudaFree(deviceDimensionality));

    // # 10. Scarto dei punti che non sono stati spinti sulla superficie della sphere mesh (dimensionality != -1)
    // # ovvero punti esterni alla sphere mesh o interni che non sono stati spinti fuori
    outPoints.clear();
    for (size_t i = 0; i < numberOfPoints; i++)
    {
        if (hostDimensionality[i] != -1)
            continue;
        outPoints.emplace_back(hostPositions[i], hostNormals[i], -1);
    }

    printf("Sono stati ottenuti %zu punti sui %zu richiesti\n", outPoints.size(), numberOfPoints);

    // # 11. TODO: Controllo di essere arrivato al numero di punti desiderato
    // # se non ci sono arrivato, riavvio creazione punti con un certo numero da definire (metà? Tenendo conto del numero di punti scartati?)
    // # per questo punto servirà refactoring profondo del ciclo do/while che andrà inserito in una funzione dedicata (con anche allocazione e distruzione memoria)

    // # 12. Eliminazione memoria allocata su host
    delete[] hostPositions;
    delete[] hostNormals;
    delete[] hostDimensionality;

    CHECK(cudaEventRecord(events[10], 0));
    CHECK(cudaEventSynchronize(events[10]));

    printf("L'esecuzione dell'algoritmo (compresa la gestione della memoria) e' durata %f millisecondi\n", computeTime(events[0], events[10]));

    for (auto &event : events)
    {
        CHECK(cudaEventDestroy(event));
    }
}