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

#include <ctime>


using glm::vec3;
using std::array;

typedef unsigned long ulong;

__device__ const float GPU_EPSILON = 0.0001f;

// 2137 MS PER COMPLICATED.SM A 10000000 PUNTI

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

__global__ void initGenerators(curandState_t *states, uint chunkSize, uint start) {
    const long tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= chunkSize)
        return;
    curand_init(tid + start, 0, 0, &states[tid]);
}

__global__ void generateRandomPointsInsideSphere(vec3 sphereCenter, float sphereRadius, glm::vec3 *pointsPositions, curandState_t *states, uint chunkSize)
{
    const long tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= chunkSize)
        return;

    curandStateXORWOW_t *state = &states[tid];

    // random 3D direction
    // NOTE: 2 *n - 1 shifts the interval from 0, 1 to -1, 1
    vec3 direction = glm::normalize(vec3(2.f * curand_uniform(state) - 1.f, 2.f * curand_uniform(state) - 1.f, 2.f * curand_uniform(state) - 1.f));

    // curand_uniform return number in (0,1], multiplied by radius return a value in (0, radius] so inside the sphere
    float extent = curand_uniform(state) * sphereRadius;

    pointsPositions[tid] = vec3(sphereCenter + direction * extent);
}

__global__ void pushOutsideSphere(glm::vec3 *positions, glm::vec3 *normals, int *dimensionalities, Sphere s0, uint chunkSize)
{
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= chunkSize)
        return;

    int dimensionality = dimensionalities[tid];

    if (dimensionality == -2)
        return;

    glm::vec3 pos = positions[tid];
    glm::vec3 CtoPos = pos - s0.center;
    const float CtoPossqrd = glm::dot(CtoPos, CtoPos);

    float sphereRadius = s0.radius;
    // pos is outside sphere
    if (CtoPossqrd > sphereRadius * sphereRadius - GPU_EPSILON)
    {
        // se il punto aveva dimensionalità -1, non dire nulla perché non sai se è interno a un'altra primitiva, ma solo che è esterno a questa
        // se il punto aveva dimensionalità != -1, non fare nulla perchè serve mantenerne l'informazione
        return;
    }

    // if we are here, pos is inside the sphere
    dimensionalities[tid] = 0;
    CtoPos = glm::normalize(CtoPos);
    positions[tid] = s0.center + sphereRadius * CtoPos;
    normals[tid] = CtoPos;
}

__device__ void pushOutsideCapsuloid(int tid, glm::vec3 *positions, glm::vec3 *normals, int *dimensionalities, const Sphere &A, const Sphere &B, float factor, glm::vec3 &BminusA)
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

__global__ void pushOutsideCapsuloidKernel(glm::vec3 *positions, glm::vec3 *normals, int *dimensionalities, Sphere s0, Sphere s1, float factor, glm::vec3 S0toS1, uint chunkSize)
{
    const long tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= chunkSize)
        return;

    int dimensionality = dimensionalities[tid];

    if (dimensionality == -2)
        return;

    pushOutsideCapsuloid(tid, positions, normals, dimensionalities, s0, s1, factor, S0toS1);
}

__global__ void pushOutsideSphereTriangle(glm::vec3 *positions, glm::vec3 *normals, int *dimensionalities, Sphere s0, Sphere s1, Sphere s2, glm::mat3 upperProjMatrix, glm::mat3 lowerProjMatrix , glm::vec3 planeN, uint chunkSize)
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

    cudaMallocHost(&hostPositions, coordinatesBytes);
    cudaMallocHost(&hostNormals, coordinatesBytes);
    cudaMallocHost(&tempDimensionalities, dimensionalityBytes);

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
    Sphere *deviceSpheres;
    Capsuloid *deviceCapsuloids;
    SphereTriangle *deviceSphereTriangles;

    CHECK(cudaEventRecord(events[2]));

    CHECK(cudaMalloc((void **)&devicePositions, coordinatesBytes));
    CHECK(cudaMalloc((void **)&deviceNormals, coordinatesBytes));
    CHECK(cudaMalloc((void **)&deviceDimensionalities, dimensionalityBytes));

    CHECK(cudaMalloc((void **)&deviceSpheres, sphereMesh.spheres.size() * sizeof(Sphere)));
    CHECK(cudaMalloc((void **)&deviceCapsuloids, sphereMesh.capsuloids.size() * sizeof(Capsuloid)));
    CHECK(cudaMalloc((void **)&deviceSphereTriangles, sphereMesh.sphereTriangles.size() * sizeof(SphereTriangle)));

    CHECK(cudaMemcpy(deviceSpheres, sphereMesh.spheres.data(), sphereMesh.spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceCapsuloids, sphereMesh.capsuloids.data(), sphereMesh.capsuloids.size() * sizeof(Capsuloid), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceSphereTriangles, sphereMesh.sphereTriangles.data(), sphereMesh.sphereTriangles.size() * sizeof(SphereTriangle), cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(events[3]));
    CHECK(cudaEventSynchronize(events[3]));

    printf("Allocati %lu bytes in memoria device in %f millisecondi...\n", coordinatesBytes + dimensionalityBytes, computeTime(events[2], events[3]));

    const int numStreams = 2;
    cudaStream_t streams[numStreams];
    for (size_t i = 0; i < numStreams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    int subBlockSize = 1024;
    uint chunkSize = (numberOfPoints / numStreams) + 1;
    int subGrid((chunkSize / subBlockSize) + 1);
    int subBlock(subBlockSize);

    float bsRadius = sphereMesh.boundingSphere.radius;
    vec3 bsCenter = sphereMesh.boundingSphere.center;

    int blockSize = 1024;
    int grid((numberOfPoints / blockSize) + 1);
    int block(blockSize);
    for (int i = 0; i < numStreams; i++) initGenerators<<<subGrid, subBlock, 0, streams[i]>>>(&devStates[chunkSize*i], chunkSize, chunkSize * i);

    uint iteration = 0U;

    // LOOP START
    while (outPoints.size() < numberOfPoints)
    {
        printf("Inizio %ua iterazione: %u/%u punti\n", iteration++, outPoints.size(), numberOfPoints);
        CHECK(cudaMemset(deviceDimensionalities, -1, numberOfPoints * sizeof(int)));

        for (int i = 0; i < numStreams; i++) generateRandomPointsInsideSphere<<<subGrid, subBlock, 0, streams[i]>>>(bsCenter, bsRadius, &devicePositions[chunkSize * i], &devStates[chunkSize*i], chunkSize);
        checkError(cudaGetLastError());


        const uint singletonStart = 0;
        const uint capsuloidStart = singletonStart + sphereMesh.singletons.size();
        const uint triangleStart = capsuloidStart + sphereMesh.capsuloids.size();
        const uint maxUniqueIdx = triangleStart + sphereMesh.sphereTriangles.size();
        const uint maxTries = 5U;

        for (size_t i = 0; i < numStreams; i++)
        {
            for (uint tries = 0; tries < maxTries; tries++)
            {
                // Primitives loop
                for (size_t uniqueIdx = 0; uniqueIdx < maxUniqueIdx; uniqueIdx++)
                {
                    if (uniqueIdx >= singletonStart && uniqueIdx < capsuloidStart)
                    {
                        pushOutsideSphere<<<subGrid, subBlock, 0, streams[i]>>>(&devicePositions[chunkSize * i], &deviceNormals[chunkSize * i], &deviceDimensionalities[chunkSize * i], sphereMesh.spheres[sphereMesh.singletons[uniqueIdx - singletonStart]], chunkSize);
                    }
                    else if (uniqueIdx >= capsuloidStart && uniqueIdx < triangleStart)
                    {
                        Capsuloid &caps = sphereMesh.capsuloids.at(uniqueIdx - capsuloidStart);

                        pushOutsideCapsuloidKernel<<<subGrid, subBlock, 0, streams[i]>>>(&devicePositions[chunkSize * i], &deviceNormals[chunkSize * i], &deviceDimensionalities[chunkSize * i], sphereMesh.spheres[caps.s0], sphereMesh.spheres[caps.s1], caps.factor, caps.S0toS1, chunkSize);
                    }
                    else if (uniqueIdx >= triangleStart)
                    {
                        SphereTriangle &st = sphereMesh.sphereTriangles.at(uniqueIdx - triangleStart);
                        pushOutsideSphereTriangle<<<subGrid, subBlock, 0, streams[i]>>>(&devicePositions[chunkSize * i], &deviceNormals[chunkSize * i], &deviceDimensionalities[chunkSize * i], sphereMesh.spheres[st.s0], sphereMesh.spheres[st.s1], sphereMesh.spheres[st.s2], st.upperProjMatrix, st.lowerProjMatrix, st.planeN, chunkSize);
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

        printf("Copia dati da device a host...\n");
        CHECK(cudaEventRecord(events[8]));
        for (size_t i = 0; i < numStreams - 1; i++)
        {
            CHECK(cudaMemcpyAsync(&hostPositions[chunkSize * i], &devicePositions[chunkSize * i], chunkSize * sizeof(vec3), cudaMemcpyDeviceToHost, streams[i]));
            CHECK(cudaMemcpyAsync(&hostNormals[chunkSize * i], &deviceNormals[chunkSize * i], chunkSize * sizeof(vec3), cudaMemcpyDeviceToHost, streams[i]));
            CHECK(cudaMemcpyAsync(&tempDimensionalities[chunkSize * i], &deviceDimensionalities[chunkSize * i], chunkSize * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
        }
        uint cumChunkSize = chunkSize * (numStreams - 1);
        uint lastChunk = numberOfPoints - cumChunkSize;
        CHECK(cudaMemcpyAsync(&hostPositions[cumChunkSize], &devicePositions[cumChunkSize], lastChunk * sizeof(vec3), cudaMemcpyDeviceToHost, streams[numStreams - 1]));
        CHECK(cudaMemcpyAsync(&hostNormals[cumChunkSize], &deviceNormals[cumChunkSize], lastChunk * sizeof(vec3), cudaMemcpyDeviceToHost, streams[numStreams - 1]));
        CHECK(cudaMemcpyAsync(&tempDimensionalities[cumChunkSize], &deviceDimensionalities[cumChunkSize], lastChunk * sizeof(int), cudaMemcpyDeviceToHost, streams[numStreams - 1]));

        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(events[9]));
        CHECK(cudaEventSynchronize(events[9]));

        printf("Copia dati da device a host TERMINATA in %f millisecondi\n", computeTime(events[8], events[9]));
        // # 8. Scarto dei punti che non sono stati spinti sulla superficie della sphere mesh (dimensionality != -1)
        // # ovvero punti esterni alla sphere mesh o interni che non sono stati spinti fuori
         for (size_t i = 0; i < numberOfPoints; i++)
        {
             if (tempDimensionalities[i] == -2)
                continue;
            outPoints.emplace_back(hostPositions[i], hostNormals[i], tempDimensionalities[i]);
        }
    }

    // # 7. Eliminazione memoria allocata sul device
    CHECK(cudaFree(devicePositions));
    CHECK(cudaFree(deviceNormals));
    CHECK(cudaFree(devStates));
    CHECK(cudaFree(deviceDimensionalities));
    for (size_t i = 0; i < numStreams ; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }

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
    cudaFreeHost(hostPositions);
    cudaFreeHost(hostNormals);
    cudaFreeHost(tempDimensionalities);
}