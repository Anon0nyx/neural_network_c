#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>
#include <errno.h>

#include "simple_neural_network.c"

double _weight = 0.5;
double _input = 0.5;
double expected_value = 0.8;
double step_amount = 0.001;

int main() {

  brute_force_learning(_input, _weight, expected_value, step_amount, 1000);

  return 0;
}
