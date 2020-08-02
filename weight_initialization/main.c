#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>
#include <errno.h>

#include "simple_neural_network.c"

#define NUM_OF_FEATURES 2
#define NUM_OF_EXAMPLES 3

#define NUM_OF_HID_NODES 3
#define NUM_OF_OUT_NODES 1

/* Hours of workout */
double x1[NUM_OF_EXAMPLES] = {2,5,1};
double _x1[NUM_OF_EXAMPLES];

/* Hours of rest */
double x2[NUM_OF_EXAMPLES] = {8,5,8};
double _x2[NUM_OF_EXAMPLES];

/*Muscle gain data*/
double y[NUM_OF_EXAMPLES] = {200, 90, 190};
double _y[NUM_OF_EXAMPLES];

double syn0[NUM_OF_HID_NODES][NUM_OF_FEATURES];

/*Hidden layer to output layer weights buffer*/
double syn1[NUM_OF_OUT_NODES][NUM_OF_HID_NODES];
int main() {

    weight_random_init(NUM_OF_HID_NODES, NUM_OF_FEATURES, syn0);
    weight_random_init(NUM_OF_OUT_NODES, NUM_OF_HID_NODES, syn1);

    /* Synapse 0 weights*/
    printf("Synapse 0 Weights : \n");
    for (int i=0; i<NUM_OF_HID_NODES;i++) {
        for (int j=0;j<NUM_OF_FEATURES;j++) {
            printf("%f\t", syn0[i][j]);
        }
        printf("\n");
    }

    /* Synapse 1 weights*/
    printf("Synapse 1 Weights : \n");
    for (int i=0;i<NUM_OF_OUT_NODES;i++) {
        for (int j=0;j<NUM_OF_HID_NODES;j++) {
            printf("%f\t", syn1[i][j]);
        }
        printf("\n");
    }

    return 0;
}
