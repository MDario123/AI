#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<random>
#include<chrono>

#define ran (long double)(int)rng() / (1ll << 31)
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
const int threads = 256;

#define F first
#define S second

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__global__ void grand(double* out, int sz, int chance, unsigned int seed, unsigned int seed2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz) {
        if (((seed * (i / sz) + (i % sz) * seed2) / (1 << 16)) % chance == 0) {
            out[i] = (double)(int)(seed * (i / sz) + (i % sz) * seed2) / (1ll << 31);
        }
    }
}
__global__ void VectorScalarMultiplication(double* vector, int size, double scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        vector[i] *= scalar;
    }
}
__global__ void VectorVectorPointAddition(double* vectorx, double* vectory, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        vectorx[i] += vectory[i];
    }
}
__global__ void VectorVectorPointMultiplication(double* vectorx, double* vectory, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        vectorx[i] *= vectory[i];
    }
}
__global__ void VectorElementSum(double* vector, int size, int stride) {
    int i = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    for (int t = 1; t < 64; t = (t << 1) + 1) {
        if ((i & t) == 0 && (i + t)*stride < size)vector[i*stride] += vector[(i + t)*stride];
    }
}
__global__ void Normalize(double* vector, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (vector[i] <= 0)vector[i] = 0;
        else vector[i] = 1;
    }
}
double VectorAddition(double* vector, int size) {
    double ans;
    
    for (int i = 1; i < size; i*=64) {
        VectorElementSum << <((size/(2*i))+threads-1)/threads, threads >> > (vector, size, i);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&ans, vector, sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(vector);
    
    return ans;
}