#pragma once
#include "AI.h"

AI GeneticTraining(const int aiAmount, int topai, double* data, int datasz, double* answer, vector<int> aiInitializer, int epochs) {
    //Variable declaration
    AI* current = new AI[aiAmount];
    for (int i = 0; i < aiAmount; i++) {
        current[i] = AI(aiInitializer);
    }
    double topscore = INFINITY;
    AI topAI = AI(aiInitializer);
    pair<double, int>* tosort = new pair<double, int>[aiAmount];

    //Training
    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "Epoch: " << epoch << "\n";
        //Evaluating the cost function
        for (int i = 0; i < aiAmount; i++) {
            tosort[i].S = i;
            tosort[i].F = current[i].solve(data, answer, datasz);
        }
        sort(tosort, tosort + aiAmount);
        //Storing the best
        if (tosort[0].F < topscore) {
            topAI.copy(current[tosort[0].S]);
            topscore = tosort[0].F;
        }
        cout << tosort[0].F << "\n";
        //Reproducing
        for (int i = topai; i < aiAmount; i++) {
            int x = rng() % topai, y = rng() % topai;
            current[tosort[i].S].add(current[tosort[x].S], current[tosort[y].S]);
        }
        //Mutating
        for (int i = 0; i < aiAmount; i++) {
            current[i].mutate(10);
        }
    }

    //Clean memory
    for (int i = 0; i < aiAmount; i++) current[i].destroy();
    free(current);
    free(tosort);

    return topAI;
}