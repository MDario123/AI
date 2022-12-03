#include "AI.h"
#define F first
#define S second
int main()
{
    AI current[aiAmount];
    for (int i = 0; i < aiAmount; i++) {
        current[i] = AI(aiInitializer);
    }
    
    double topscore = INFINITY;
    AI topAI = AI(aiInitializer);

    double* data;
    cudaMallocManaged((void**)&data, datasz * sizeof(double));
    for (int i = 0; i < datasz; i++)data[i] = i;

    double* answer = new double[datasz * aiInitializer.back()];
    for (int i = 0; i < datasz * aiInitializer.back(); i++)answer[i] = i;

    for (int epoch = 1; epoch <= 1000; epoch++) {
        cout << "Epoch: " << epoch << "\n";
        pair<double, int> tosort[aiAmount];
        for (int i = 0; i < aiAmount; i++) {
            tosort[i].S = i;
            tosort[i].F = 0;
            double* out = current[i].solve(data, datasz);
            for (int t = 0; t < datasz * aiInitializer.back(); t++) {
                tosort[i].F += (out[t] - answer[t]) * (out[t] - answer[t]);
            }

            free(out);
        }
        sort(tosort, tosort + aiAmount);
        if (tosort[0].F < topscore) {
            topAI.copy(current[tosort[0].S]);
            topscore = tosort[0].F;
        }
        cout << tosort[0].F << "\n";
        for (int i = logai; i < aiAmount; i++) {
            int x = rng() % logai, y = rng() % logai;
            current[tosort[i].S].add(current[tosort[x].S], current[tosort[y].S]);
        }
        for (int i = 0; i < aiAmount; i++) {
            current[i].mutate(10);
        }
    }

    cudaDeviceReset();
    return 0;
}