#include <stdio.h>
#include "simple_neural_network.c"

#define NUM_OF_INPUTS 3
double temperatures[] = {12,23,50,16,-10};
double humidity[] = {60, 67, 50, 65, 63};
double air_quality[] = {60,47,25,76,34};

double weight[] = {-2,2,1};

int main() {
  
  double training_ex_1[3] = {temperatures[0], humidity[0], air_quality[0]};

  printf("The prediction from the first training example is %.2f\n", mul_input_single_out_nn(training_ex_1, weight, NUM_OF_INPUTS));
}

