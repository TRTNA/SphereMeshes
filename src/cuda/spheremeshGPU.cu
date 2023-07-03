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

struct Points
{
    float *posX;
    float *posY;
    float *posZ;
    float *normX;
    float *normY;
    float *normZ;
    int *dimensionality;
};

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

__global__ void checkDimensionality(Points points, uint numPoints, int offset)
{
    const long tid = offset + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numPoints)
        return;
    if (points.dimensionality[tid] == -1)
        points.dimensionality[tid] = -2;
}

__global__ void initGenerators(curandState_t *states, uint chunkSize, uint start)
{
    const long tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= chunkSize)
        return;
    curand_init(tid + start, 0, 0, &states[tid]);
}

__global__ void generateRandomPointsInsideSphere(vec3 sphereCenter, float sphereRadius, Points points, curandState_t *states, uint numPoints, int offset)
{
    const long tid = offset + threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= numPoints)
        return;

    curandStateXORWOW_t *state = &states[tid];

    // random 3D direction
    // NOTE: 2 *n - 1 shifts the interval from 0, 1 to -1, 1
    vec3 direction = glm::normalize(vec3(2.f * curand_uniform(state) - 1.f, 2.f * curand_uniform(state) - 1.f, 2.f * curand_uniform(state) - 1.f));

    // curand_uniform return number in (0,1], multiplied by radius return a value in (0, radius] so inside the sphere
    float extent = curand_uniform(state) * sphereRadius;

    vec3 result(sphereCenter + direction * extent);
    points.posX[tid] = result.x;
    points.posY[tid] = result.y;
    points.posZ[tid] = result.z;
}

__global__ void pushOutsideSphere(Points points, Sphere s0, uint numPoints, int offset)
{
    const long tid = offset + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numPoints)
        return;

    int dimensionality = points.dimensionality[tid];

    if (dimensionality == -2)
        return;

    glm::vec3 pos(points.posX[tid], points.posY[tid], points.posZ[tid]);
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
    points.dimensionality[tid] = 0;
    CtoPos = glm::normalize(CtoPos);
    vec3 result = s0.center + sphereRadius * CtoPos;

    points.posX[tid] = result.x;
    points.posY[tid] = result.y;
    points.posZ[tid] = result.z;
    points.normX[tid] = CtoPos.x;
    points.normY[tid] = CtoPos.y;
    points.normZ[tid] = CtoPos.z;
}

__device__ void pushOutsideCapsuloid(int tid, Points points, const Sphere &A, const Sphere &B, float factor, glm::vec3 &BminusA)
{
    glm::vec3 pos(points.posX[tid], points.posY[tid], points.posZ[tid]);

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
    points.dimensionality[tid] = (int)dim;
    const vec3 normal = glm::normalize(CtoPos);

    vec3 result = C + interpRadius * normal;

    points.posX[tid] = result.x;
    points.posY[tid] = result.y;
    points.posZ[tid] = result.z;
    points.normX[tid] = normal.x;
    points.normY[tid] = normal.y;
    points.normZ[tid] = normal.z;
}

__global__ void pushOutsideCapsuloidKernel(Points points, Sphere s0, Sphere s1, float factor, glm::vec3 S0toS1, uint numPoints, int offset)
{
    const long tid = offset + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numPoints)
        return;
    int dimensionality = points.dimensionality[tid];

    if (dimensionality == -2)
        return;

    pushOutsideCapsuloid(tid, points, s0, s1, factor, S0toS1);
}

__global__ void pushOutsideSphereTriangle(Points points, Sphere s0, Sphere s1, Sphere s2, glm::mat3 upperProjMatrix, glm::mat3 lowerProjMatrix, glm::vec3 planeN, uint pointsNum, int offset)
{
    const long tid = offset + blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= pointsNum)
        return;

    int dimensionality = points.dimensionality[tid];
    if (dimensionality == -2)
        return;

    glm::vec3 pos(points.posX[tid], points.posY[tid], points.posZ[tid]);

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
        pushOutsideCapsuloid(tid, points, s0, s1, factor, s1minuss0);
        return;
    }
    if (c < 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V1V2
        glm::vec3 s2minuss1 = s2.center - s1.center;
        float factor = (s2.radius - s1.radius) / glm::dot(s2minuss1, s2minuss1);
        pushOutsideCapsuloid(tid, points, s1, s2, factor, s2minuss1);
        return;
    }
    if (a < 0.0f)
    {
        // PUSH OUTSIDE CAPSULE V0V2
        glm::vec3 s2minuss0 = s2.center - s0.center;
        float factor = (s2.radius - s0.radius) / glm::dot(s2minuss0, s2minuss0);
        pushOutsideCapsuloid(tid, points, s0, s2, factor, s2minuss0);
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
        points.dimensionality[tid] = 2;
        glm::vec3 normal = glm::normalize(CtoPos);
        vec3 result = C + interpRadius * normal;

        points.posX[tid] = result.x;
        points.posY[tid] = result.y;
        points.posZ[tid] = result.z;
        points.normX[tid] = normal.x;
        points.normY[tid] = normal.y;
        points.normZ[tid] = normal.z;
    }
}

void createSphereMeshGPU(SphereMesh &sphereMesh, uint numberOfPoints, std::vector<DimensionalityPoint> &outPoints)
{
    printf("Starting...\n");
    cudaSetDevice(0);

    // Inizializzazione memoria host
    printf("Inizializzazione memoria host...\n");
    ulong pointsBytes = numberOfPoints * (sizeof(float) * 6 + sizeof(int));

    Points hostPoints;

    CHECK(cudaMallocHost((void **)&hostPoints.posX, sizeof(float) * numberOfPoints));
    CHECK(cudaMallocHost((void **)&hostPoints.posY, sizeof(float) * numberOfPoints));
    CHECK(cudaMallocHost((void **)&hostPoints.posZ, sizeof(float) * numberOfPoints));

    CHECK(cudaMallocHost((void **)&hostPoints.normX, sizeof(float) * numberOfPoints));
    CHECK(cudaMallocHost((void **)&hostPoints.normY, sizeof(float) * numberOfPoints));
    CHECK(cudaMallocHost((void **)&hostPoints.normZ, sizeof(float) * numberOfPoints));

    CHECK(cudaMallocHost((void **)&hostPoints.dimensionality, sizeof(int) * numberOfPoints));


    // Inizializzazione memoria device
    printf("Inizializzazione memoria device...\n");

    curandState *devStates;
    cudaMalloc((void **)&devStates, numberOfPoints * sizeof(curandState));

    Points devicePoints;
    CHECK(cudaMalloc((void **)&devicePoints.posX, sizeof(float) * numberOfPoints));
    CHECK(cudaMalloc((void **)&devicePoints.posY, sizeof(float) * numberOfPoints));
    CHECK(cudaMalloc((void **)&devicePoints.posZ, sizeof(float) * numberOfPoints));

    CHECK(cudaMalloc((void **)&devicePoints.normX, sizeof(float) * numberOfPoints));
    CHECK(cudaMalloc((void **)&devicePoints.normY, sizeof(float) * numberOfPoints));
    CHECK(cudaMalloc((void **)&devicePoints.normZ, sizeof(float) * numberOfPoints));

    CHECK(cudaMalloc((void **)&devicePoints.dimensionality, sizeof(int) * numberOfPoints));

    const int numStreams = 1;
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
    for (int i = 0; i < numStreams; i++)
        initGenerators<<<subGrid, subBlock, 0, streams[i]>>>(&devStates[chunkSize * i], chunkSize, chunkSize * i);

    uint iteration = 0U;

    // LOOP START
    while (outPoints.size() < numberOfPoints)
    {
        printf("Inizio %ua iterazione: %u/%u punti\n", iteration++, outPoints.size(), numberOfPoints);
        CHECK(cudaMemset(devicePoints.dimensionality, -1, numberOfPoints * sizeof(int)));
        cudaDeviceSynchronize();

        for (int i = 0; i < numStreams; i++)
        {
            generateRandomPointsInsideSphere<<<subGrid, subBlock, 0, streams[i]>>>(bsCenter, bsRadius, devicePoints, devStates, numberOfPoints,  i * chunkSize);
            checkError(cudaGetLastError());
        }

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
                        pushOutsideSphere<<<subGrid, subBlock, 0, streams[i]>>>(devicePoints, sphereMesh.spheres[sphereMesh.singletons[uniqueIdx - singletonStart]], numberOfPoints, i * chunkSize);

                    }
                    else if (uniqueIdx >= capsuloidStart && uniqueIdx < triangleStart)
                    {
                        Capsuloid &caps = sphereMesh.capsuloids.at(uniqueIdx - capsuloidStart);

                        pushOutsideCapsuloidKernel<<<subGrid, subBlock, 0, streams[i]>>>(devicePoints, sphereMesh.spheres[caps.s0], sphereMesh.spheres[caps.s1], caps.factor, caps.S0toS1, numberOfPoints,  i * chunkSize);

                    }
                    else if (uniqueIdx >= triangleStart)
                    {
                        SphereTriangle &st = sphereMesh.sphereTriangles.at(uniqueIdx - triangleStart);
                        pushOutsideSphereTriangle<<<subGrid, subBlock, 0, streams[i]>>>(devicePoints, sphereMesh.spheres[st.s0], sphereMesh.spheres[st.s1], sphereMesh.spheres[st.s2], st.upperProjMatrix, st.lowerProjMatrix, st.planeN, numberOfPoints,  i * chunkSize);

                    }
                }
                checkError(cudaGetLastError());
                if (tries == 0)
                {
                    // è il primo tentativo, se un punto è rimasto a -1 allora è esterno, va scartato
                    //  chiamata a kernel che setta i punti a -1 su -2 (verrano ignorati negli altri kernel)
                    checkDimensionality<<<subGrid, subBlock, 0, streams[i]>>>(devicePoints, numberOfPoints,  i * chunkSize);
                    checkError(cudaGetLastError());
                }
            }
        }


        printf("Copia dati da device a host...\n");
        long coordinateBytes = sizeof(float) * numberOfPoints;
        CHECK(cudaMemcpy(hostPoints.posX, devicePoints.posX, coordinateBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hostPoints.posY, devicePoints.posY, coordinateBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hostPoints.posZ, devicePoints.posZ, coordinateBytes, cudaMemcpyDeviceToHost));

        CHECK(cudaMemcpy(hostPoints.normX, devicePoints.normX, coordinateBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hostPoints.normY, devicePoints.normY, coordinateBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hostPoints.normZ, devicePoints.normZ, coordinateBytes, cudaMemcpyDeviceToHost));

        CHECK(cudaMemcpy(hostPoints.dimensionality, devicePoints.dimensionality, sizeof(int) * numberOfPoints, cudaMemcpyDeviceToHost));

        // # Scarto dei punti che non sono stati spinti sulla superficie della sphere mesh (dimensionality != -1)
        // # ovvero punti esterni alla sphere mesh o interni che non sono stati spinti fuori
        for (size_t i = 0; i < numberOfPoints; i++)
        {
            if (hostPoints.dimensionality[i] == -2)
                continue;
            outPoints.emplace_back(vec3(hostPoints.posX[i], hostPoints.posY[i], hostPoints.posZ[i]),
                                   vec3(hostPoints.normX[i], hostPoints.normY[i], hostPoints.normZ[i]),
                                   hostPoints.dimensionality[i]);
        }
    }

    // Eliminazione memoria allocata sul device
    CHECK(cudaFree(devicePoints.posX));
    CHECK(cudaFree(devicePoints.posY));
    CHECK(cudaFree(devicePoints.posZ));
    CHECK(cudaFree(devicePoints.normX));
    CHECK(cudaFree(devicePoints.normY));
    CHECK(cudaFree(devicePoints.normZ));
    CHECK(cudaFree(devicePoints.dimensionality));

    CHECK(cudaFree(devStates));
    for (size_t i = 0; i < numStreams; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    // Eliminazione memoria allocata su host
    CHECK(cudaFreeHost(hostPoints.posX));
    CHECK(cudaFreeHost(hostPoints.posY));
    CHECK(cudaFreeHost(hostPoints.posZ));
    CHECK(cudaFreeHost(hostPoints.normX));
    CHECK(cudaFreeHost(hostPoints.normY));
    CHECK(cudaFreeHost(hostPoints.normZ));
    CHECK(cudaFreeHost(hostPoints.dimensionality));
}