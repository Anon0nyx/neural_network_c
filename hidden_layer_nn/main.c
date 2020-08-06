#include <stdio.h>
#include <stdlib.h>

#define SAD_PRED_IDX 0
#define SICK_PRED_IDX 1
#define ACTIVE_PRED_IDX 2

#define IN_LEN 3
#define OUT_LEN 3
#define HID_LEN 3

#include "simple_neural_network.c"

int main() {
  double hidden_pred_vector[HID_LEN];
  double prediction[3];
  double in_to_hid_weight[OUT_LEN][IN_LEN] = {  {-2, 9.5, 2.01}, // hid[0]
					                            {-0.8, 7.2, 6.3}, // hid[1]
					                            {-0.5, 0.45, 0.9} // hid[2]
                                             };

  double hid_to_out_weight[OUT_LEN][HID_LEN] = { {-1.0, 1.15, 0.11}, // SAD
						 {-.18, 0.15, -0.01}, // SICK
						 {0.25, -0.25, -0.1} // ACTIVE
  };
  
  //                       temp hum air_q
  double _input[IN_LEN] = { 30, 87, 110 };

  hidden_layer_nn(_input, IN_LEN, HID_LEN, in_to_hid_weight, OUT_LEN, hid_to_out_weight, prediction, hidden_pred_vector);

  printf("Sad prediction : %f\n", prediction[SAD_PRED_IDX]);
  printf("Sick prediction : %f\n", prediction[SICK_PRED_IDX]);
  printf("Active prediction : %f\n", prediction[ACTIVE_PRED_IDX]);
  
  return 0;
}
  
