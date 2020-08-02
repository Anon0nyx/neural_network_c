double single_in_single_out(double _input, double _weight) {

  return (_input * _weight);
}

double weighted_sum(double * _input, double * _weight, int LEN) {

  double output;
  for (int i=0; i<LEN; i++) {
    output += (_input[i] * _weight[i]);
  }

  return output;
}

double mul_in_single_out_nn(double * _input, double * _weight, int LEN) {

  double output;
  return weighted_sum(_input, _weight, LEN);
}

void elementwise_multiply(double _input_scalar, double * _weight_vector, double * _output_vector, int LEN) {

  // The purpose of this function is to multiply each weight by the single input to determine the output values
  for (int i=0; i<LEN; i++) {
    _output_vector[i] = (_input_scalar * _weight_vector[i]);
  }
}

single_in_mul_out_nn(double _input_scalar, double * _weight_vector, double * _output_vector, int LEN) {

  elementwise_multiply(_input_scalar, _weight_vector, _output_vector, LEN);
}
