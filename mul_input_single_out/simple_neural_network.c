#include <stdio.h>

double single_in_single_out(double _input, double _weight) {
  
  return (_input * _weight);
}

double weighted_sum(double * _input, double * _weight, int LEN) {

  double output;

  for (int i=0; i<LEN; i++) {

    output += _input[i] * _weight[i];
  }

  return output;
}

double mul_input_single_out_nn(double * _input, double * _weight, int LEN) {

  double predicted_value = weighted_sum(_input, _weight, LEN);
  return predicted_value;
}
