#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "nn_utilities.c"

typedef struct {
    float *weights;                     /*All weights*/
    float *hid_layer_to_out_weights;    /*hidden layer to output layer weights*/
    float *b;                           /*biases*/
    float *hid_layers;                  /*hidden layer*/
    float *out_layer;                   /*output layer*/
    int num_biases;                     /*number of biases*/
    int num_weights;                    /*number of weights*/
    int num_inputs;                     /*number of inputs*/
    int num_hid_layers;                 /*number of hidden neurons*/
    int num_outputs;                    /*number of outputs*/
}NeuralNetwork_Type;

static float toterr(const float *const target,const float *const out_layer, const int size);

// One line functions
static float err(const float actual, const float predicted) { return 0.5f * (actual - predicted) * (actual - predicted); }
static float pd_err(const float a, const float b) { return a - b; }
static float sigmoid_activation(const float a) { return 1.0f / (1.0f + expf(-a)); } // Sigmoid activation function
static float pd_sigmoid_act(const float a) { return a * (1.0f - a); } // Partial derivative of sigmoid activation function
static float frand() { return rand() / (float)RAND_MAX; }


static void fprop(const NeuralNetwork_Type network, const float *const input) {

    /*Hidden layer neuron values*/
    for(int i=0; i < network.num_hid_layers; i++) {
        float sum = 0.0f;
        for(int j=0; j < network.num_inputs; j++) {
            sum += input[j] * network.weights[ i * network.num_inputs + j ];
        }
        network.hid_layers[i] =  sigmoid_activation(sum + network.b[0]);
    }

    /*Output layer neuron values*/
    for(int i=0; i < network.num_outputs; i++) {
        float sum = 0.0f;
        for(int j=0; j < network.num_hid_layers; j++) {
            sum += network.hid_layers[j] * network.hid_layer_to_out_weights[ i * network.num_hid_layers + j ];
        }
        network.out_layer[i] = sigmoid_activation(sum+network.b[1]);
    }
}

static void bprop(const NeuralNetwork_Type network,
                  const float *const input,
                  const float *const target,
                  float rate) {

    for(int i=0; i < network.num_hid_layers; i++) {
        float sum = 0.0f;
        for(int j=0; j < network.num_outputs; j++) {
            const float a = pd_err(network.out_layer[j],target[j]);
            const float b = pd_sigmoid_act(network.out_layer[j]);

            sum += a * b * network.hid_layer_to_out_weights[ j * network.num_hid_layers + i ];
            network.hid_layer_to_out_weights[ j * network.num_hid_layers + i ] -= rate * a * b * network.hid_layers[i];
        }
        for(int j=0; j < network.num_inputs; j++) {
            network.weights[ i * network.num_inputs + j ] -= rate * sum * pd_sigmoid_act(network.hid_layers[i]) * input[j];
        }
    }
}

static void wbrand(const NeuralNetwork_Type network) {
     for(int i=0; i < network.num_weights; i++) {
         network.weights[i] = frand() - 0.5f;
     }
     for(int i=0; i < network.num_biases; i++) {
         network.b[i] = frand() - 0.5f;
     }
}

float *NNpredict(const NeuralNetwork_Type network, const float *input ) {

    fprop(network,input);
    return network.out_layer;
}

NeuralNetwork_Type NNbuild(const int num_inputs, const int num_hid_layers, const int num_outputs) {

    NeuralNetwork_Type network;
    network.num_biases = 2;                                                              /*number of biases*/
    network.num_weights =  num_hid_layers * (num_inputs +num_outputs);                   /*number of weights*/
    network.weights =  (float *)calloc(network.num_weights, sizeof(*network.weights));    /*All weights*/
    network.hid_layer_to_out_weights =  network.weights + num_hid_layers *num_inputs;    /*hidden layer to output layer weights*/
    network.b = (float *)calloc(network.num_biases, sizeof(*network.b));                  /*biases*/
    network.hid_layers = (float *)calloc(num_hid_layers, sizeof(*network.hid_layers));    /*hidden layer*/
    network.out_layer = (float *)calloc(num_outputs, sizeof(*network.out_layer));         /*output layer*/
    network.num_inputs = num_inputs;                                                     /*number of inputs*/
    network.num_hid_layers =  num_hid_layers;                                            /*number of hidden neurons*/
    network.num_outputs = num_outputs;                                                   /*number of outputs*/
    wbrand(network);

    return network;
}

void NNsave(const NeuralNetwork_Type network, const char *path) {

    FILE *const file =  fopen(path, "weights");
    /*Save the header*/
    fprintf(file, "%d %d %d\n", network.num_inputs, network.num_hid_layers, network.num_outputs);

    /*Save the biases*/
    for(int i =0; i < network.num_biases; i++) {
        fprintf(file,"%f\n", (double)network.b[i]);
    }

    /*Save the weights*/
    for(int i=0; i < network.num_weights; i++) {
        fprintf(file,"%f\n", (double)network.weights[i]);
    }
    fclose(file);
}

NeuralNetwork_Type NNload(const char *path) {

    FILE *const file = fopen(path, "r");
    int num_inputs = 0;
    int num_hid_layers = 0;
    int num_outputs = 0;

    /*Load the header*/
    fscanf(file, "%d %d %d\n", &num_inputs, &num_hid_layers, &num_outputs);

    const NeuralNetwork_Type network = NNbuild(num_inputs, num_hid_layers, num_outputs);

    /*Load the biases*/
    for(int i=0; i < network.num_biases; i++) {
        fscanf(file,"%f\n", &network.b[i]);
    }
    /*Load the weights*/
    for(int i=0; i < network.num_weights; i++) {
        fscanf(file, "%f\n", &network.weights[i]);
    }
    fclose(file);
    return network;
}

float NNtrain(const NeuralNetwork_Type network, const float *input, const float *target, float rate) {

     fprop(network, input);
     bprop(network, input, target, rate);

     return toterr(target, network.out_layer, network.num_outputs);
}

void NNprint(const float *arr, const int size) {

    double max =  0.0f;
    int idx;
    for(int i=0; i < size; i++) {
        printf("%f ", (double)arr[i]);
        if(arr[i] > max) {
            idx = i;
            max = arr[i];
        }
    }
    printf("\n");
    printf("The number is : %d\n", idx);
}

void NNfree(const NeuralNetwork_Type network) {

    free(network.weights);
    free(network.b);
    free(network.hid_layers);
    free(network.out_layer);
}

static float toterr(const float *const target, const float *const out_layer, const int size) {

    float sum = 0.0f;
    for(int i=0; i < size; i++) {
        sum += err(target[i], out_layer[i]);
    }
    return sum;
}

Data build(const char *path, const int num_inputs, const int num_outputs) {

    FILE *file = fopen(path, "r");
    if(file == NULL) {
        printf("Could not open %s.\n", path);
        printf("Dataset does not exist.\n");
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(num_inputs, num_outputs, rows);

    for(int row=0; row < rows; row++) {
        char *line = readln(file);
        parse(data, line, row);
        free(line);
    }

    fclose(file);
    return data;
}
