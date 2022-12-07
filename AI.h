#pragma once
using namespace std;
#include "Parallel_Basic.h"
#include <iostream>
#include<vector>
#include<random>
#include<chrono>

#define ran (long double)(int)rng() / (1ll << 31)
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

__global__ void LayerEval(double* out, double* in, double* factors, double* bias, int sz, int insz, int outsz, bool b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz * outsz) {
        int x = i / outsz;
        int y = i - x * outsz;
        double tem = 0;
        for (int t = 0; t < insz; t++) {
            tem += in[x * insz + t] * factors[y * insz + t];
        }
        tem += bias[y];
        if (b && tem < 0)out[i] = 0;
        else out[i] = tem;
    }
}
__global__ void grand(double* out, int sz, int chance, unsigned int seed, unsigned int seed2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz) {
        if (((seed + i  * seed2)/(1<<16)) % chance == 0) {
            out[i] = (double)(int)(seed + (sz + i) * seed2) / (1ll << 31);
        }
    }
}
struct AI {
    int layers;
    int* dim;
    double** bias;
    double** factors;
    AI() {}
    AI(vector<int> v) {
        layers = v.size();
        dim = new int[layers];
        bias = new double* [layers];
        factors = new double* [layers];
        dim[0] = v[0];
        for (int i = 1; i < layers; i++) {
            dim[i] = v[i];
            gpuErrchk(cudaMalloc((void**)&bias[i], dim[i] * sizeof(double)));
            gpuErrchk(cudaMalloc((void**)&factors[i], dim[i] * dim[i - 1] * sizeof(double)));
        }
        mutate(1);
    }
    void add(AI& ai, AI& ai1) {
        for (int i = 1; i < layers; i++) {
            if (rng() & 1) {
                gpuErrchk(cudaMemcpy(bias[i], ai.bias[i], dim[i] * sizeof(double), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemcpy(factors[i], ai.factors[i], dim[i] * dim[i - 1] * sizeof(double), cudaMemcpyDeviceToDevice));
            }
            else {
                gpuErrchk(cudaMemcpy(bias[i], ai1.bias[i], dim[i] * sizeof(double), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemcpy(factors[i], ai1.factors[i], dim[i] * dim[i - 1] * sizeof(double), cudaMemcpyDeviceToDevice));
            }
        }
    }
    void copy(AI& ai) {
        for (int i = 1; i < layers; i++) {
            gpuErrchk(cudaMemcpy(bias[i], ai.bias[i], dim[i] * sizeof(double), cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(factors[i], ai.factors[i], dim[i] * dim[i - 1] * sizeof(double), cudaMemcpyDeviceToDevice));
        }
    }
    //Partially randomizes the weights in the AI, each weight has a chance(chance) of being randomized
    void mutate(const int& chance) {
        for (int i = 1; i < layers; i++) {
            grand << <(dim[i] + threads - 1) / threads, threads >> > (bias[i], dim[i], chance, rng(), rng());
            grand << <(dim[i] * dim[i - 1] + threads - 1) / threads, threads >> > (factors[i], dim[i] * dim[i - 1], chance, rng(), rng());
        }
        cudaDeviceSynchronize();
    }
    double* solve(double* data, int size) {
        double* in = data;
        for (int i = 1; i < layers; i++) {
            double* out;
            gpuErrchk(cudaMalloc((void**)&out, size * dim[i] * sizeof(double)));
            LayerEval << <(size * dim[i] + threads - 1) / threads, threads >> > (out, in, factors[i], bias[i], size, dim[i-1] , dim[i], 1);
            gpuErrchk(cudaDeviceSynchronize());
            if(i>1)cudaFree(in);
            gpuErrchk(cudaDeviceSynchronize());
            in = out;
            out = nullptr;
        }
        double* value = new double[size * dim[layers - 1]];
        gpuErrchk(cudaMemcpy(value, in, size * dim[layers - 1] * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(in);
        return value;
    }
};
