#include <cuda/spheremeshGPU.h>
#include <spheremeshes/point.h>
#include <spheremeshes/spheremesh.h>
#include <spheremeshes/sphere.h>
#include <spheremeshes/capsuloid.h>

#include <utils/random.h>
#include <utils/aabb.h>

#include <glm/glm.hpp>

#include <stdio.h>

#include <cmath>

#include <array>

#include <curand_kernel.h>

using glm::vec3;
using std::array;

typedef unsigned long ulong;

__device__ const float GPU_EPSILON = 0.0001f;

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



__global__ void checkDimensionality(int *dimensionalities, uint chunkSize)
{
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= chunkSize)
        return;
    if (dimensionalities[tid] == -1)
        dimensionalities[tid] = -2;
}

__global__ void generateRandomPointsInsideSphere(vec3 sphereCenter, float sphereRadius, glm::vec3 *pointsPositions, curandState_t *states, uint chunkSize)
{
    const long tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= chunkSize)
        return;
    curand_init(tid, 0, 0, &states[tid]);

    // random 3D direction
    // NOTE: 2 *n - 1 shifts the interval from 0, 1 to -1, 1
    vec3 direction = glm::normalize(vec3(2.f * curand_uniform(&states[tid]) - 1.f, 2.f * curand_uniform(&states[tid]) - 1.f, 2.f * curand_uniform(&states[tid]) - 1.f));

    // curand_uniform return number in (0,1], multiplied by radius return a value in (0, radius] so inside the sphere
    float extent = curand_uniform(&states[tid]) * sphereRadius;

    pointsPositions[tid] = vec3(sphereCenter + direction * extent);
}

__global__ void pushOutsideSphere(glm::vec3 *positions, glm::vec3 *normals, int *dimensionalities, Sphere sphere, uint chunkSize)
{
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= chunkSize)
        return;

    int dimensionality = dimensionalities[tid];

    if (dimensionality == -2)
        return;

    glm::vec3 pos = positions[tid];
    glm::vec3 CtoPos = pos - sphere.center;
    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    // pos is outside sphere
    if (CtoPossqrd > sphere.radius * sphere.radius - GPU_EPSILON)
    {
        // se il punto aveva dimensionalità -1, non dire nulla perché non sai se è interno a un'altra primitiva, ma solo che è esterno a questa
        // se il punto aveva dimensionalità != -1, non fare nulla perchè serve mantenerne l'informazione
        return;
    }

    // if we are here, pos is inside the sphere
    dimensionalities[tid] = 0;
    CtoPos = glm::normalize(CtoPos);
    positions[tid] = sphere.center + sphere.radius * CtoPos;
    normals[tid] = CtoPos;
}

__device__ void pushOutsideCapsuloid(int tid, glm::vec3 *positions, glm::vec3 *normals, int *dimensionalities, Sphere &A, Sphere &B, float factor, glm::vec3 &BminusA)
{
    vec3 pos = positions[tid];

    float k = glm::dot(pos - A.center, BminusA) / glm::dot(BminusA, BminusA);
    vec3 fakeC = A.center + k * BminusA;
    float d = length(fakeC - pos);

    k += (factor * d);

    const float clampedK = glm::clamp(k, 0.0f, 1.0f);

    const vec3 C = A.center + clampedK * BminusA;

    const vec3 CtoPos = pos - C;

    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    const float interpRadius = A.radius * (1.0f - clampedK) + B.radius * clampedK;

    // pos is outside the capsule, dimensionality is -1 (not pushed out)
    if (CtoPossqrd > interpRadius * interpRadius - GPU_EPSILON)
    {
        // se il punto aveva dimensionalità -1, non dire nulla perché non sai se è interno a un'altra primitiva, ma solo che è esterno a questa
        // se il punto aveva dimensionalità != -1, non fare nulla perchè serve mantenerne l'informazione
        return;
    }

    // if we are here, pos is inside the capsule
    // dimensionality depends on K value
    // if clampedK == k then pos is inside the cylinder, so dimensionality = 1
    // else pos is inside one of the spheres, so dimensionality = 0
    bool dim = k == clampedK;
    dimensionalities[tid] = (int)dim;
    const vec3 normal = glm::normalize(CtoPos);

    positions[tid] = C + interpRadius * normal;
    normals[tid] = normal;
}

__global__ void pushOutsideCapsuloidKernel(glm::vec3 *positions, glm::vec3 *normals, int *dimensionalities, Sphere A, Sphere B, float factor, glm::vec3 BminusA, uint chunkSize)
{
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= chunkSize)
        return;

    int dimensionality = dimensionalities[tid];

    if (dimensionality == -2)
        return;

    pushOutsideCapsuloid(tid, positions, normals, dimensionalities, A, B, factor, BminusA);
}

__global__ void pushOutsideSphereTriangle(glm::vec3 *positions, glm::vec3 *normals, int *dimensionalities, Sphere s0, Sphere s1, Sphere s2, glm::mat4 upperProjMatrix, glm::mat4 lowerProjMatrix, glm::vec3 planeN, uint chunkSize)
{
    const long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= chunkSize)
        return;

    int dimensionality = dimensionalities[tid];
    if (dimensionality == -2)
        return;

    glm::vec3 pos = positions[tid];

    const vec3 q = pos - s0.center;
    float d, a, b, c;
    const glm::mat3 projMatrix = glm::dot(q, planeN) < 0 ? lowerProjMatrix : upperProjMatrix;
    const vec3 res = projMatrix * q;
    d = res.z;
    a = res.x;
    b = res.y;
    c = (1.0f - a - b);

    if (b < 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V0V1
        glm::vec3 s1minuss0 = s1.center - s0.center;
        float factor = (s1.radius - s0.radius) / glm::dot(s1minuss0, s1minuss0);
        pushOutsideCapsuloid(tid, positions, normals, dimensionalities, s0, s1, factor, s1minuss0);
        return;
    }
    if (c < 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V1V2
        glm::vec3 s2minuss1 = s2.center - s1.center;
        float factor = (s2.radius - s1.radius) / glm::dot(s2minuss1, s2minuss1);
        pushOutsideCapsuloid(tid, positions, normals, dimensionalities, s1, s2, factor, s2minuss1);
        return;
    }
    if (a < 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V0V2
        glm::vec3 s2minuss0 = s2.center - s0.center;
        float factor = (s2.radius - s0.radius) / glm::dot(s2minuss0, s2minuss0);
        pushOutsideCapsuloid(tid, positions, normals, dimensionalities, s0, s2, factor, s2minuss0);
        return;
    }

    // PUSH OUTSIDE TRIANGLE
    if (a < 1.0f && b < 1.0f && c < 1.0f)
    {
        vec3 C = c * s0.center + a * s1.center + b * s2.center;
        float interpRadius = c * s0.radius + a * s1.radius + b * s2.radius;
        vec3 CtoPos = pos - C;
        if (d > interpRadius - GPU_EPSILON)
        {
            // se il punto aveva dimensionalità -1, non dire nulla perché non sai se è interno a un'altra primitiva, ma solo che è esterno a questa
            // se il punto aveva dimensionalità != -1, non fare nulla perchè serve mantenerne l'informazione
            return;
        }
        dimensionalities[tid] = 2;
        glm::vec3 normal = glm::normalize(CtoPos);
        positions[tid] = C + interpRadius * normal;
        normals[tid] = normal;
    }
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
    ulong coordinatesBytes = numberOfPoints * sizeof(float) * 3;
    ulong dimensionalityBytes = numberOfPoints * sizeof(int);

    glm::vec3 *hostPositions = (glm::vec3 *)malloc(coordinatesBytes);
    glm::vec3 *hostNormals = (glm::vec3 *)malloc(coordinatesBytes);

    int *tempDimensionalities = (int *)malloc(dimensionalityBytes);

    curandState *devStates;
    cudaMalloc((void **)&devStates, numberOfPoints * sizeof(curandState));

    CHECK(cudaEventRecord(events[1], 0));
    // wait until the stop event completes
    CHECK(cudaEventSynchronize(events[1]));
    printf("Allocati %lu bytes in memoria host in %f millisecondi...\n", coordinatesBytes + dimensionalityBytes, computeTime(events[0], events[1]));

    // # 2. Inizializzazione memoria device
    printf("Inizializzazione memoria device...\n");

    int *deviceDimensionalities;
    glm::vec3 *devicePositions, *deviceNormals;
    CHECK(cudaEventRecord(events[2]));

    CHECK(cudaMalloc((void **)&devicePositions, coordinatesBytes));
    CHECK(cudaMalloc((void **)&deviceNormals, coordinatesBytes));

    CHECK(cudaMalloc((void **)&deviceDimensionalities, dimensionalityBytes));
    CHECK(cudaMemset(deviceDimensionalities, -1, numberOfPoints * sizeof(int)));

    CHECK(cudaEventRecord(events[3]));
    CHECK(cudaEventSynchronize(events[3]));

    printf("Allocati %lu bytes in memoria device in %f millisecondi...\n", coordinatesBytes + dimensionalityBytes, computeTime(events[2], events[3]));

    // # 3. creazione punti random dentro la bounding sphere della sphere mesh

    printf("Generazione posizioni random su GPU...\n");
    cudaStream_t streams[10];
    for (size_t i = 0; i < 10; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    int subBlockSize = 256;
    uint chunkSize = (numberOfPoints / 10) + 1;
    dim3 subGrid((chunkSize / subBlockSize) + 1);
    dim3 subBlock(subBlockSize);

    float bsRadius = sphereMesh.boundingSphere.radius;
    vec3 bsCenter = sphereMesh.boundingSphere.center;

    CHECK(cudaEventRecord(events[4]));
    for (size_t i = 0; i < 10; i++)
    {
        generateRandomPointsInsideSphere<<<subGrid, subBlock, 0, streams[i]>>>(bsCenter, bsRadius, &devicePositions[chunkSize * i], &devStates[chunkSize * i], chunkSize);
    }
    CHECK(cudaDeviceSynchronize());
    checkError(cudaGetLastError());
    CHECK(cudaEventRecord(events[5]));
    CHECK(cudaEventSynchronize(events[5]));
    printf("Generate %u posizioni in memoria device in %f millisecondi...\n", numberOfPoints, computeTime(events[4], events[5]));

    // # 4. Creazione contesto loop
    const uint maxTries = 5U;

    CHECK(cudaEventRecord(events[6]));

    const uint singletonStart = 0;
    const uint edgeStart = singletonStart + sphereMesh.singletons.size();
    const uint triangleStart = edgeStart + sphereMesh.capsuloids.size();
    const uint maxUniqueIdx = triangleStart + sphereMesh.sphereTriangles.size();

    // # 5. Loop di creazione dei punti
    for (size_t i = 0; i < 10; i++)
    {
        for (uint tries = 0; tries < maxTries; tries++)
        {
            // Primitives loop
            for (size_t uniqueIdx = 0; uniqueIdx < maxUniqueIdx; uniqueIdx++)
            {
                if (uniqueIdx >= singletonStart && uniqueIdx < edgeStart)
                {
                    pushOutsideSphere<<<subGrid, subBlock, 0, streams[i]>>>(&devicePositions[chunkSize * i], &deviceNormals[chunkSize * i], &deviceDimensionalities[chunkSize * i], sphereMesh.spheres.at(sphereMesh.singletons.at(uniqueIdx)), chunkSize);
                }
                else if (uniqueIdx >= edgeStart && uniqueIdx < triangleStart)
                {
                    Capsuloid &caps = sphereMesh.capsuloids.at(uniqueIdx - edgeStart);

                    pushOutsideCapsuloidKernel<<<subGrid, subBlock, 0, streams[i]>>>(&devicePositions[chunkSize * i], &deviceNormals[chunkSize * i], &deviceDimensionalities[chunkSize * i], sphereMesh.spheres.at(caps.s0), sphereMesh.spheres.at(caps.s1), caps.factor, caps.S0toS1, chunkSize);
                }
                else if (uniqueIdx >= triangleStart)
                {
                    SphereTriangle &st = sphereMesh.sphereTriangles.at(uniqueIdx - triangleStart);
                    pushOutsideSphereTriangle<<<subGrid, subBlock, 0, streams[i]>>>(&devicePositions[chunkSize * i], &deviceNormals[chunkSize * i], &deviceDimensionalities[chunkSize * i], sphereMesh.spheres.at(st.vertices[0]), sphereMesh.spheres.at(st.vertices[1]), sphereMesh.spheres.at(st.vertices[2]), st.upperProjMatrix, st.lowerProjMatrix, st.planeN, chunkSize);
                }
            }
            checkError(cudaGetLastError());
            if (tries == 0)
            {
                // è il primo tentativo, se un punto è rimasto a -1 allora è esterno, va scartato
                //  chiamata a kernel che setta i punti a -1 su -2 (verrano ignorati negli altri kernel)
                checkDimensionality<<<subGrid, subBlock, 0, streams[i]>>>(&deviceDimensionalities[chunkSize * i], chunkSize);
                checkError(cudaGetLastError());
            }
        }
    }

    CHECK(cudaDeviceSynchronize());

    // # Esce dal ciclo quanto tutti i punti sono o esterni alla sphere mesh o spinti sulla superficie
    CHECK(cudaEventRecord(events[7]));
    CHECK(cudaEventSynchronize(events[7]));

    printf("Creazione punti TERMINATA in %f millisecondi\n", computeTime(events[6], events[7]));

    // # 6. Copia da memoria device a memoria host (funziona così?)
    printf("Copia dati da device a host...\n");
    CHECK(cudaEventRecord(events[8]));
    for (size_t i = 0; i < 9; i++)
    {
        CHECK(cudaMemcpyAsync(&hostPositions[chunkSize * i], &devicePositions[chunkSize * i], chunkSize * sizeof(vec3), cudaMemcpyDeviceToHost, streams[i]));
        CHECK(cudaMemcpyAsync(&hostNormals[chunkSize * i], &deviceNormals[chunkSize * i], chunkSize * sizeof(vec3), cudaMemcpyDeviceToHost, streams[i]));
        CHECK(cudaMemcpyAsync(&tempDimensionalities[chunkSize * i], &deviceDimensionalities[chunkSize * i], chunkSize * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }
    uint cumChunkSize = chunkSize * 9;
    uint lastChunk = numberOfPoints - cumChunkSize;
    CHECK(cudaMemcpyAsync(&hostPositions[cumChunkSize], &devicePositions[cumChunkSize], lastChunk * sizeof(vec3), cudaMemcpyDeviceToHost, streams[9]));
    CHECK(cudaMemcpyAsync(&hostNormals[cumChunkSize], &deviceNormals[cumChunkSize], lastChunk * sizeof(vec3), cudaMemcpyDeviceToHost, streams[9]));
    CHECK(cudaMemcpyAsync(&tempDimensionalities[cumChunkSize], &deviceDimensionalities[cumChunkSize], lastChunk * sizeof(int), cudaMemcpyDeviceToHost, streams[9]));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(events[9]));
    CHECK(cudaEventSynchronize(events[9]));

    printf("Copia dati da device a host TERMINATA in %f millisecondi\n", computeTime(events[8], events[9]));

    // # 7. Eliminazione memoria allocata sul device
    CHECK(cudaFree(devicePositions));
    CHECK(cudaFree(deviceNormals));
    CHECK(cudaFree(devStates));
    CHECK(cudaFree(deviceDimensionalities));
    for (size_t i = 0; i < 10; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    // # 8. Scarto dei punti che non sono stati spinti sulla superficie della sphere mesh (dimensionality != -1)
    // # ovvero punti esterni alla sphere mesh o interni che non sono stati spinti fuori
    outPoints.clear();
    for (size_t i = 0; i < numberOfPoints; i++)
    {
        if (tempDimensionalities[i] == -2)
            continue;
        outPoints.emplace_back(hostPositions[i], hostNormals[i], tempDimensionalities[i]);
    }

    printf("Sono stati ottenuti %zu punti sui %zu richiesti\n", outPoints.size(), numberOfPoints);

    // # 9. TODO: Controllo di essere arrivato al numero di punti desiderato
    // # se non ci sono arrivato, riavvio creazione punti con un certo numero da definire (metà? Tenendo conto del numero di punti scartati?)
    // # per questo punto servirà refactoring profondo del ciclo do/while che andrà inserito in una funzione dedicata (con anche allocazione e distruzione memoria)

    CHECK(cudaEventRecord(events[10], 0));
    CHECK(cudaEventSynchronize(events[10]));

    printf("L'esecuzione dell'algoritmo (compresa la gestione della memoria) e' durata %f millisecondi\n", computeTime(events[0], events[10]));

    for (auto &event : events)
    {
        CHECK(cudaEventDestroy(event));
    }

    // # 10. Eliminazione memoria allocata su host
    delete[] hostPositions;
    delete[] hostNormals;
    delete[] tempDimensionalities;
}