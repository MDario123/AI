#include "AI.h"
#define F first
#define S second
int main()
{
    AI current[aiAmount];
    for (int i = 0; i < aiAmount; i++) {
        current[i] = AI(aiInitializer);
    }
    double* data;
    cudaMallocManaged((void**)&data, datasz * sizeof(double));
    for (int i = 0; i < datasz; i++)data[i] = i;

    double* answer = new double[datasz * aiInitializer.back()];
    for (int i = 0; i < datasz * aiInitializer.back(); i++)answer[i] = i;

    for (int epoch = 1; epoch <= 1000; epoch++) {
        pair<double, int> tosort[aiAmount];
        for (int i = 0; i < aiAmount; i++) {
            tosort[i].S = i;
            tosort[i].F = 0;
            double* out = current[i].solve(data, datasz);
            for (int t = 0; t < datasz * aiInitializer.back(); t++) {
                tosort[i].F += (out[i] - answer[i]) * (out[i] - answer[i]);
            }
            free(out);
        }

    }

    cudaDeviceReset();
    return 0;
}