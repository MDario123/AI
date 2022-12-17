#include "Training.h"

const int inputDataSize = 152640;

vector<int> aiInitializer{ 0, 64, 64, 1 };
const int topai = 3;
const int aiAmount = 100;
int main()
{
    freopen("trainingData.txt", "r", stdin);

    //input
    
    //[inputDataSize][7][11]
    double*** f = new double** [inputDataSize];
    for (int i = 0; i < inputDataSize; i++) {
        f[i] = new double* [7];
        for (int t = 0; t < 7; t++) {
            f[i][t] = new double[11];
            cin >> f[i][t][0];
        }
        f[i][0][0] /= 60;
    }
    
    int stride = 0;
    for (int i = 0; i < 11; i++) {
        stride += (1 << i) * (i + 1);
        aiInitializer[0] += (i + 1);
    }

    //prebuild power-of-2 states
    
    for (int i = 1; i < 11; i++) {
        for (int t = 0; t + (1 << i) < inputDataSize; t++) {
            f[t][0][i] = 0;
            f[t][1][i] = (f[t][1][i - 1] + f[t + (1 << (i - 1))][1][i - 1]) / 2;
            f[t][2][i] = (f[t][2][i - 1] + f[t + (1 << (i - 1))][2][i - 1]) / 2;
            f[t][3][i] = max(f[t][3][i - 1], f[t + (1 << (i - 1))][3][i - 1]);
            f[t][4][i] = min(f[t][4][i - 1], f[t + (1 << (i - 1))][4][i - 1]);
            f[t][5][i] = (f[t][5][i - 1] + f[t + (1 << (i - 1))][5][i - 1]) / 2;
            f[t][6][i] = (f[t][6][i - 1] + f[t + (1 << (i - 1))][6][i - 1]) / 2;
        }
    }
    
    //generate "answers"
    
    double* answer = new double[(inputDataSize - stride - 60)];
    double best = 0;
    for (int i = stride; i + 60 < inputDataSize; i++) {
        answer[i - stride] = f[i + 60][1][0] / f[i][1][0] * 0.98 - 1;
        if (answer[i - stride] > 0)best += answer[i - stride];
    }
    cout << best << "\n";
    //regularize

    double* mi = new double[7];
    double* ma = new double[7];
    for (int i = 0; i < 7; i++) {
        mi[i] = INFINITY;
        ma[i] = -INFINITY;
    }

    for (int i = 0; i < inputDataSize; i++) {
        for (int t = 0; t < 7; t++) {
            mi[t] = min(mi[t], f[i][t][0]);
            ma[t] = max(ma[t], f[i][t][0]);
        }
    }

    double width[7], center[7];
    for (int i = 0; i < 7; i++) {
        width[i] = ma[i] - mi[i];
        center[i] = (ma[i]+mi[i])/2;
    }

    for (int i = 0; i < inputDataSize; i++) {
        for (int t = 0; t < 7; t++) {
            f[i][t][0] = (f[i][t][0]-center[t])/width[t];
        }
    }

    //format

    aiInitializer[0] = aiInitializer[0] * 6 + 1;
    double* data = new double[aiInitializer[0] * (inputDataSize - stride - 60)];
    for (int i = stride; i + 60 < inputDataSize; i++) {
        data[(i - stride) * aiInitializer[0]] = f[i - 1][0][0];
        for (int t = 0; t < 11; t++) {
            int partialStride = 0;
            for (int e = 0; e <= t; e++) {
                partialStride += (1 << t);
                data[(i - stride) * aiInitializer[0] + (t * (t + 1) / 2 + e) * 6 + 1] = f[i - partialStride][1][t];
                data[(i - stride) * aiInitializer[0] + (t * (t + 1) / 2 + e) * 6 + 2] = f[i - partialStride][2][t];
                data[(i - stride) * aiInitializer[0] + (t * (t + 1) / 2 + e) * 6 + 3] = f[i - partialStride][3][t];
                data[(i - stride) * aiInitializer[0] + (t * (t + 1) / 2 + e) * 6 + 4] = f[i - partialStride][4][t];
                data[(i - stride) * aiInitializer[0] + (t * (t + 1) / 2 + e) * 6 + 5] = f[i - partialStride][5][t];
                data[(i - stride) * aiInitializer[0] + (t * (t + 1) / 2 + e) * 6 + 6] = f[i - partialStride][6][t];
            }
        }
    }
    
    //copying data to gpu

    double* gpuData;
    cudaMalloc((void**)&gpuData, (aiInitializer[0] * (inputDataSize - stride - 60)) * sizeof(double));
    gpuErrchk(cudaMemcpy(gpuData, data, (aiInitializer[0] * (inputDataSize - stride - 60)) * sizeof(double), cudaMemcpyHostToDevice));
    double* gpuAnswer;
    cudaMalloc((void**)&gpuAnswer, (inputDataSize - stride - 60) * sizeof(double));
    gpuErrchk(cudaMemcpy(gpuAnswer, answer, (inputDataSize - stride - 60) * sizeof(double), cudaMemcpyHostToDevice));

    //releasing memory

    for (int i = 0; i < inputDataSize; i++) {
        for (int t = 0; t < 7; t++) {
            free(f[i][t]);
        }
        free(f[i]);
    }
    free(f);
    free(ma);
    free(mi);
    free(data);

    //training

    GeneticTraining(aiAmount, topai, gpuData, inputDataSize - stride - 60, gpuAnswer, aiInitializer, 100);

    cudaFree(gpuData);
    cudaDeviceReset();
    return 0;
}