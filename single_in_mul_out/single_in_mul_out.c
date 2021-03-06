#include <stdio.h>
#include "simple_neural_network.c"

#define Happy 2

#define TEMPERATURE_PREDICTION_IDX 0
#define HUMIDITY_PREDICTION_IDX 1
#define AIR_QUALITY_PREDICTION_IDX 2
#define LEN 3

int main() {

  double _weight[3] = { 25, 54, 123 };
  double prediction[3];

  single_in_mul_out_nn(Happy, _weight, prediction, LEN);
  printf("Prediction for temperaturee : %.2f\n", prediction[TEMPERATURE_PREDICTION_IDX]);
  printf("Prediction for humidity : %.2f\n", prediction[HUMIDITY_PREDICTION_IDX]);
  printf("Prediction for air-quality: %.2f\n", prediction[AIR_QUALITY_PREDICTION_IDX]);
  
  return 0;
}
  
