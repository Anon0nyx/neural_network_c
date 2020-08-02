#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>
#include <errno.h>

#include "simple_neural_network.c"

#define NUM_OF_FEATURES 2
#define NUM_OF_EXAMPLES 3

/* Hours of workout */
double x1[NUM_OF_EXAMPLES] = {2,5,1};
double _x1[NUM_OF_EXAMPLES];

/* Hours of rest */
double x2[NUM_OF_EXAMPLES] = {8,5,8};
double _x2[NUM_OF_EXAMPLES];

/*Muscle gain data*/
double y[NUM_OF_EXAMPLES] = {200, 90, 190};
double _y[NUM_OF_EXAMPLES];

int main() {

    normalize_data(x1, _x1, NUM_OF_EXAMPLES);
    normalize_data(x2, _x2, NUM_OF_EXAMPLES);
    normalize_data(y, _y, NUM_OF_EXAMPLES);

    printf("Raw x1 data : \n\r");
    for (int i=0;i<NUM_OF_EXAMPLES;i++) {
        printf("%f\n", x1[i]);
    }
    printf("Normalized x1 data : \n\r");
    for (int i=0;i<NUM_OF_EXAMPLES;i++) {
        printf("%f\n", _x1[i]);
    }

    printf("Raw x2 data : \n\r");
    for (int i=0;i<NUM_OF_EXAMPLES;i++) {
        printf("%f\n", x2[i]);
    }
    printf("Normalized x2 data : \n\r");
    for (int i=0;i<NUM_OF_EXAMPLES;i++) {
        printf("%f\n", _x2[i]);
    }

    printf("Raw y data : \n\r");
    for (int i=0;i<NUM_OF_EXAMPLES;i++) {
        printf("%f\n", y[i]);
    }
    printf("Normalized y data : \n\r");
    for (int i=0;i<NUM_OF_EXAMPLES;i++) {
        printf("%f\n", _y[i]);
    }
    return 0;
}
