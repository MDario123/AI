#include "Training.h"

vector<int> aiInitializer{ 1, 4, 4, 1 };
const int datasz = 1000;
const int topai = 10;
const int aiAmount = 1024;

int main()
{
    double* data;
    cudaMallocManaged((void**)&data, datasz * sizeof(double));
    for (int i = 0; i < datasz; i++)data[i] = i;

    double* answer;
    cudaMallocManaged((void**)&answer, datasz * sizeof(double));
    for (int i = 0; i < datasz * aiInitializer.back(); i++)answer[i] = -i;

    GeneticTraining(aiAmount, topai, data, datasz, answer, aiInitializer, 10);

    cudaFree(data);
    cudaFree(answer);
    _sleep(5000);
    cudaDeviceReset();
    return 0;
}