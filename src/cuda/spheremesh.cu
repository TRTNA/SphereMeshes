#include <cuda/spheremesh.h>
#include <spheremeshes/point.h>

#include <stdio.h>

typedef unsigned long ulong;

float computeTime(cudaEvent_t &e1, cudaEvent_t &e2)
{
    float time;
    cudaEventElapsedTime(&time, e1, e2);
    return time;
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

__global__ void testKernel(void)
{
    printf("funza");
}

void createSphereMesh(SphereMesh &sphereMesh, uint numberOfPoints, std::vector<Point> &outPoints)
{
    printf("Starting...\n");
    cudaSetDevice(0);
      cudaDeviceProp dProp;
	cudaGetDeviceProperties(&dProp, 0);
    printf("Major %d Minor %d\n", dProp.major, dProp.minor);
    cudaEvent_t allAppStart, allAppEnd, hostMemEnd, hostMemStart, deviceMemStart, deviceMemEnd, memCpyHTDStart, memCpyHTDEnd;
    cudaEvent_t memCpyDTHStart, memCpyDTHEnd;
    CHECK(cudaEventCreate(&allAppStart));
    CHECK(cudaEventCreate(&allAppEnd));
    CHECK(cudaEventCreate(&hostMemStart));
    CHECK(cudaEventCreate(&hostMemEnd));
    CHECK(cudaEventCreate(&deviceMemStart));
    CHECK(cudaEventCreate(&deviceMemEnd));
    CHECK(cudaEventCreate(&memCpyHTDStart));
    CHECK(cudaEventCreate(&memCpyHTDEnd));
    CHECK(cudaEventCreate(&memCpyDTHStart));
    CHECK(cudaEventCreate(&memCpyDTHEnd));

    // # 1. Inizializzazione memoria host
    CHECK(cudaEventRecord(allAppStart));
    printf("Inizializzazione memoria host...\n");
    ulong pointsCoordBytes = numberOfPoints * sizeof(float);
    ulong dimensionalityBytes = numberOfPoints * sizeof(int);

    float *hostX = (float *)malloc(pointsCoordBytes);
    float *hostY = (float *)malloc(pointsCoordBytes);
    float *hostZ = (float *)malloc(pointsCoordBytes);
    int *hostDimensionality = (int *)malloc(dimensionalityBytes);
    for (size_t i = 0; i < numberOfPoints; i++)
    {
        hostDimensionality[i] = -1;
    }
    CHECK(cudaEventRecord(hostMemEnd));
    printf("Allocati %lu bytes in memoria host in %f secondi...\n", pointsCoordBytes * 3 + dimensionalityBytes, computeTime(allAppStart, hostMemEnd));

    // # 2. TODO inizializzare i punti con valori random dentro la bounding sphere della sphere mesh
    // # on GPU?

    // # 3. Inizializzazione memoria device
    printf("Inizializzazione memoria device...\n");
    float *deviceX, *deviceY, *deviceZ;
    int *deviceDimensionality;
    CHECK(cudaEventRecord(deviceMemStart));

    CHECK(cudaMalloc((void **)&deviceX, pointsCoordBytes));

    CHECK(cudaMalloc((void **)&deviceY, pointsCoordBytes));
    CHECK(cudaMalloc((void **)&deviceZ, pointsCoordBytes));
    CHECK(cudaMalloc((void **)&deviceDimensionality, dimensionalityBytes));

    CHECK(cudaEventRecord(deviceMemEnd));
    printf("Allocati %lu bytes in memoria device in %f secondi...\n", pointsCoordBytes * 3 + dimensionalityBytes, computeTime(deviceMemStart, deviceMemEnd));

    // # 4. Copia da memoria host a memoria device
    printf("Copia dati da host a device...\n");
    CHECK(cudaEventRecord(memCpyHTDStart));
    CHECK(cudaMemcpy(deviceX, hostX, pointsCoordBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceY, hostY, pointsCoordBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceZ, hostZ, pointsCoordBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceDimensionality, hostDimensionality, dimensionalityBytes, cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(memCpyHTDEnd));
    printf("Copia dati da host a device TERMINATA in %f secondi\n", computeTime(memCpyHTDStart, memCpyHTDEnd));

    // # 6. Creazione contesto loop
    const uint maxTries = 10U;
    uint tries = 0U;
    bool *allNegativeDim;
    CHECK(cudaMallocManaged(&allNegativeDim, sizeof(int)));

    // # 7. Loop di creazione dei punti
    do
    {
        printf("Tentativo %u...", tries);
        // # 7.1 TODO Chiamata al kernel di push outside
        // # pushOutside<<<grid, block>>>(devicePoints, numberOfPoints);

        // # 7.2 Sincronizzazione sul lavoro del kernel pushOutside
        printf("Attesa terminazione kernel...\n");
        dim3 grid(1);
        dim3 block(32, 32);
        testKernel<<<grid, block>>>();
        cudaError_t error = cudaGetLastError();
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error,
                cudaGetErrorString(error));
        CHECK(cudaDeviceSynchronize());

        // # 7.3 TODO Chiamata al kernel che controlla se sono tutte negative le dimensionality
        printf("Controllo dimensionalita' punti...\n");
        // # checkAllNegativeDimensionalities<<<grid, block>>>(devicePoints, numberOfPoints, allNegativeDim);
        tries++;
    } while (!(*allNegativeDim) && tries < maxTries);
    // # Esce dal ciclo quanto tutti i punti sono o esterni alla sphere mesh o spinti sulla superficie
    printf("Creazione punti terminata perche' %s\n", tries == maxTries ? "sono stati esauriti i tentativi" : "i punti sono tutti esterni o sulla superficie");

    // # 8. Copia da memoria device a memoria host (funziona così?)
    printf("Copia dati da device a host...\n");
    CHECK(cudaEventRecord(memCpyDTHStart));
    CHECK(cudaMemcpy(hostX, deviceX, pointsCoordBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostY, deviceY, pointsCoordBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostZ, deviceZ, pointsCoordBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostDimensionality, deviceDimensionality, dimensionalityBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(memCpyDTHEnd));
    printf("Copia dati da device a host TERMINATA in %f secondi\n", computeTime(memCpyDTHStart, memCpyDTHEnd));

    // # 9. Eliminazione memoria allocata sul device (memcpy è bloccante, sono sicuro che non mi serva più quando arrivo qui)
    CHECK(cudaFree(deviceX));
    CHECK(cudaFree(deviceY));
    CHECK(cudaFree(deviceZ));
    CHECK(cudaFree(deviceDimensionality));

    // # 10. Scarto dei punti che non sono stati spinti sulla superficie della sphere mesh (dimensionality != -1)
    // # ovvero punti esterni alla sphere mesh o interni che non sono stati spinti fuori
    vector<Point> points;
    for (size_t i = 0; i < numberOfPoints; i++)
    {
        if (hostDimensionality[i] != -1)
            continue;
        points.emplace_back(
            hostX[i],
            hostY[i],
            hostZ[i]);
    }

    printf("Sono stati ottenuti %u punti sui %u richiesti\n");

    // # 11. TODO: Controllo di essere arrivato al numero di punti desiderato
    // # se non ci sono arrivato, riavvio creazione punti con un certo numero da definire (metà? Tenendo conto del numero di punti scartati?)
    // # per questo punto servirà refactoring profondo del ciclo do/while che andrà inserito in una funzione dedicata (con anche allocazione e distruzione memoria)

    // # 12. Eliminazione memoria allocata su host
    delete hostX;
    delete hostY;
    delete hostZ;
    delete hostDimensionality;

    CHECK(cudaEventRecord(allAppEnd, 0));
    printf("L'esecuzione dell'algoritmo (compresa la gestione della memoria) e' durata %f secondi\n", computeTime(allAppStart, allAppEnd));

    CHECK(cudaEventDestroy(allAppStart));
    CHECK(cudaEventDestroy(allAppEnd));
}