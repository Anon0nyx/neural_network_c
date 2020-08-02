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

double raw_x[NUM_OF_FEATURES][NUM_OF_EXAMPLES] = {
                                                    { 2, 5, 1 },
                                                    { 8, 5, 8 },
                                                 };
double raw_y[1][NUM_OF_EXAMPLES] = { { 200, 90, 190 }};
/* Train x
    2  5  1  -> 2/8  5/8  1/8
    8  5  8  -> 8/8  5/8  8/8
    dimension = nx X m (num features by num examples training)
*/
double train_x[NUM_OF_FEATURES][NUM_OF_EXAMPLES];

/*Train y
    200/200  90/200  190/200
    dimensions = 1 X m (1 by num training examples)
*/
double train_y[1][NUM_OF_EXAMPLES];

/* Input layer to hidden layer weights buffer */
double syn0[NUM_OF_HID_NODES][NUM_OF_FEATURES];

/*Hidden layer to output layer weights buffer*/
double syn1[NUM_OF_HID_NODES];

double train_x_eg1[NUM_OF_FEATURES];
double train_y_eg1;
double z1_eg1[NUM_OF_HID_NODES];
double a1_eg1[NUM_OF_HID_NODES];
double z2_eg1;
double yhat_eg1;
int main() {

    /* Normalize x and y */
    normalize_data_2d(NUM_OF_FEATURES, NUM_OF_EXAMPLES, raw_x, train_x);
    normalize_data_2d(1, NUM_OF_EXAMPLES, raw_y, train_y);

    train_x_eg1[0] = train_x[0][0];
    train_x_eg1[1] = train_x[1][1];

    train_y_eg1 = train_y[0][0];

    printf("train_x_eg1 is [%f %f]", train_x_eg1[0], train_x_eg1[1]);
    printf("\n\r\n\r");

    printf("train_y_eg1 is %f", train_y_eg1);
    printf("\n\r\n\r");

    /* Initialize Synapse Zero & One (syn0 & syn1) */
    weight_random_init(NUM_OF_HID_NODES, NUM_OF_FEATURES, syn0);

    /* Synapse 0 weights */
    printf("Synapse 0 weights : \n\r");
    for (int i=0;i<NUM_OF_HID_NODES;i++) {
        for (int j=0;j<NUM_OF_FEATURES;j++) {
            printf("%f\t", syn0[i][j]);
        }
        printf("\n\r");
    }
    printf("\n\r");

    weight_random_init_1d(syn1, NUM_OF_OUT_NODES);
    /* Synapse 1 weights */
    printf("Synapse 1 weights : \n\r");
    for (int i=0;i<NUM_OF_OUT_NODES;i++) {
        printf("[%f\t%f\t%f]", syn1[0], syn1[1], syn1[2]);
    }
    printf("\n\r\n\r");

    /* Compute z1 */
    mul_in_mul_out_nn(train_x_eg1, NUM_OF_FEATURES,
                        z1_eg1, NUM_OF_HID_NODES,
                        syn0);
    printf("z1_eg1 = [%f   %f   %f]", z1_eg1[0], z1_eg1[1], z1_eg1[2]);
    printf("\n\r\n\r");

    /* Compute a1 */
    vector_sigmoid(z1_eg1, a1_eg1, NUM_OF_HID_NODES);
    printf("\n\r\n\r");

    printf("a1_eg1 = [%f   %f   %f]", a1_eg1[0], a1_eg1[1], a1_eg1[2]);
    printf("\n\r\n\r");

    /* Compute z2 */
    z2_eg1 = mul_input_single_out_nn(a1_eg1, syn1, NUM_OF_HID_NODES);
    printf("z2_eg1 : %f", z2_eg1);
    printf("\n\r\n\r");

    /* Compute yhat */
    yhat_eg1 = sigmoid(z2_eg1);
    printf("yhat_eg1 :\t%f", yhat_eg1);
    printf("\n\r\n\r");

    return 0;
}
