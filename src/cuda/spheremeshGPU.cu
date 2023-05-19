#include <cuda/spheremeshGPU.h>
#include <spheremeshes/point.h>
#include <spheremeshes/sphere.h>

#include <utils/random.h>

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

__global__ void pushOutsideSphere(float *dX, float *dY, float *dZ, int *dimensionality, float xC, float yC, float zC, float radius)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float CtoPosX = dX[tid] - xC;
    float CtoPosY = dY[tid] - yC;
    float CtoPosZ = dZ[tid] - zC;
    const float CtoPossqrd = dot(CtoPosX, CtoPosY, CtoPosZ, CtoPosX, CtoPosY, CtoPosZ);

    // pos is outside sphere
    if (CtoPossqrd > radius * radius - 0.0001f)
    {
        dimensionality[tid] = -2;
    }

    // if we are here, pos is inside the sphere
    dimensionality[tid] = -1;
    const float length = sqrt(CtoPossqrd);
    float CtoPosXNorm = CtoPosX / length;
    float CtoPosYNorm = CtoPosY / length;
    float CtoPosZNorm = CtoPosZ / length;
    dX[tid] = xC + radius * CtoPosXNorm;
    dY[tid] = yC + radius * CtoPosYNorm;
    dZ[tid] = zC + radius * CtoPosZNorm;
}

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
    ulong pointsCoordBytes = numberOfPoints * sizeof(float);
    ulong dimensionalityBytes = numberOfPoints * sizeof(int);

    float *hostX = (float *)malloc(pointsCoordBytes);
    float *hostY = (float *)malloc(pointsCoordBytes);
    float *hostZ = (float *)malloc(pointsCoordBytes);
    int *hostDimensionality = (int *)malloc(dimensionalityBytes);
    for (size_t i = 0; i < numberOfPoints; i++)
    {
    }
    CHECK(cudaEventRecord(events[1], 0));
    // wait until the stop event completes
    CHECK(cudaEventSynchronize(events[1]));
    printf("Allocati %lu bytes in memoria host in %f millisecondi...\n", pointsCoordBytes * 3 + dimensionalityBytes, computeTime(events[0], events[1]));

    // # 2. TODO inizializzare i punti con valori random dentro la bounding sphere della sphere mesh
    // # on GPU?
    for(size_t i = 0; i < numberOfPoints; i++) {
        hostX[i] = generateFloat(-2.0f, 2.0f);
        hostY[i] = generateFloat(-2.0f, 2.0f);
        hostZ[i] = generateFloat(-2.0f, 2.0f);
        hostDimensionality[i] = -1;
    }

    // # 3. Inizializzazione memoria device
    printf("Inizializzazione memoria device...\n");
    float *deviceX, *deviceY, *deviceZ;
    int *deviceDimensionality;
    CHECK(cudaEventRecord(events[2]));

    CHECK(cudaMalloc((void **)&deviceX, pointsCoordBytes));

    CHECK(cudaMalloc((void **)&deviceY, pointsCoordBytes));
    CHECK(cudaMalloc((void **)&deviceZ, pointsCoordBytes));
    CHECK(cudaMalloc((void **)&deviceDimensionality, dimensionalityBytes));

    CHECK(cudaEventRecord(events[3]));
    CHECK(cudaEventSynchronize(events[3]));

    printf("Allocati %lu bytes in memoria device in %f millisecondi...\n", pointsCoordBytes * 3 + dimensionalityBytes, computeTime(events[2], events[3]));

    // # 4. Copia da memoria host a memoria device
    printf("Copia dati da host a device...\n");
    CHECK(cudaEventRecord(events[4]));
    CHECK(cudaMemcpy(deviceX, hostX, pointsCoordBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceY, hostY, pointsCoordBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceZ, hostZ, pointsCoordBytes, cudaMemcpyHostToDevice));
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

    // TODO tbd wrt points number
    int blockSize = 1024;
    dim3 grid((numberOfPoints / blockSize) + 1);
    dim3 block(blockSize);

    // # 7. Loop di creazione dei punti
    do
    {
        printf("Tentativo %u...", tries);
        // # 7.1 TODO Chiamata al kernel di push outside
        // # pushOutside<<<grid, block>>>(devicePoints, numberOfPoints);

        // # 7.2 Sincronizzazione sul lavoro del kernel pushOutside
        printf("Attesa terminazione kernel...\n");

        pushOutsideSphere<<<grid, block>>>(deviceX, deviceY, deviceZ, deviceDimensionality, 0.0f, 0.0f, 0.0f, 2.0f);
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
    CHECK(cudaMemcpy(hostX, deviceX, pointsCoordBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostY, deviceY, pointsCoordBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostZ, deviceZ, pointsCoordBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostDimensionality, deviceDimensionality, dimensionalityBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(events[9]));
    CHECK(cudaEventSynchronize(events[9]));

    printf("Copia dati da device a host TERMINATA in %f millisecondi\n", computeTime(events[8], events[9]));

    // # 9. Eliminazione memoria allocata sul device (memcpy è bloccante, sono sicuro che non mi serva più quando arrivo qui)
    CHECK(cudaFree(deviceX));
    CHECK(cudaFree(deviceY));
    CHECK(cudaFree(deviceZ));
    CHECK(cudaFree(deviceDimensionality));

    // # 10. Scarto dei punti che non sono stati spinti sulla superficie della sphere mesh (dimensionality != -1)
    // # ovvero punti esterni alla sphere mesh o interni che non sono stati spinti fuori
    outPoints.clear();
    for (size_t i = 0; i < numberOfPoints; i++)
    {
        if (hostDimensionality[i] != -1)
            continue;
        outPoints.emplace_back(
            hostX[i],
            hostY[i],
            hostZ[i]);
    }

    printf("Sono stati ottenuti %zu punti sui %zu richiesti\n", outPoints.size(), numberOfPoints);

    // # 11. TODO: Controllo di essere arrivato al numero di punti desiderato
    // # se non ci sono arrivato, riavvio creazione punti con un certo numero da definire (metà? Tenendo conto del numero di punti scartati?)
    // # per questo punto servirà refactoring profondo del ciclo do/while che andrà inserito in una funzione dedicata (con anche allocazione e distruzione memoria)

    // # 12. Eliminazione memoria allocata su host
    delete hostX;
    delete hostY;
    delete hostZ;
    delete hostDimensionality;

    CHECK(cudaEventRecord(events[10], 0));
    CHECK(cudaEventSynchronize(events[10]));

    printf("L'esecuzione dell'algoritmo (compresa la gestione della memoria) e' durata %f millisecondi\n", computeTime(events[0], events[10]));

    for (auto &event : events)
    {
        CHECK(cudaEventDestroy(event));
    }
}