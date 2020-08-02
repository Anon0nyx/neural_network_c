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

void elementwise_multiply(double _input_scalar, double * _weight_vector, double * _output_vector, int LEN) {

  for (int i=0; i<LEN; i++) {
    _output_vector[i] = (_input_scalar * _weight_vector[i]);
  }
}

void single_in_mul_out_nn(double _input_scalar, double * _weight_vector, double * _output_vector, int LEN) {

  elementwise_multiply(_input_scalar, _weight_vector, _output_vector, LEN);
}

void matrix_vector_multiply(double * _input_vector,
			    int INPUT_LEN,
			    double * _output_vector,
			    int OUTPUT_LEN,
			    double _weight_matrix[OUTPUT_LEN][INPUT_LEN]) {
  for (int i=0; i<OUTPUT_LEN; i++) {
    for (int k=0; k<INPUT_LEN;  k++) {
      _output_vector[k] += _input_vector[i] * _weight_matrix[k][i];
    }
  }
}

void mul_in_mul_out_nn(double * _input_vector,
		       int INPUT_LEN,
		       double * _output_vector,
		       int OUTPUT_LEN,
		       double _weight_matrix[OUTPUT_LEN][INPUT_LEN]) {
  matrix_vector_multiply(_input_vector, INPUT_LEN, _output_vector, OUTPUT_LEN, _weight_matrix);
}
