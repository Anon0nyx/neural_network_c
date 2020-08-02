#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fenv.h>
#include <errno.h>

double single_in_single_out(double _input, double _weight) {

    return (_input * _weight);
}

double weighted_sum(double * _input, double * _weight, int LEN) {

    double _output;

    for (int i=0; i<LEN; i++) {

          _output += _input[i] * _weight[i];
    }

    return  _output;
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

void hidden_layer_nn(double * _input_vector,
		     int INPUT_LEN,
		     int HIDDEN_LEN,
		     double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
		     int OUTPUT_LEN,
		     double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN],
		     double * _output_vector,
		     double * hidden_pred_vector) {

    matrix_vector_multiply(_input_vector, INPUT_LEN, hidden_pred_vector, OUTPUT_LEN, in_to_hid_weights);
    matrix_vector_multiply(hidden_pred_vector, HIDDEN_LEN, _output_vector, OUTPUT_LEN, hid_to_out_weights);
}

double find_error(double _input, double _weight, double expected_value) {

    return powf(((_input * _weight) - expected_value), 2);
}

double find_error_simple(double yhat, double y) {

    return powf((yhat - y), 2);
}

void brute_force_learning(double _input, double _weight, double expected_value, double step_amount, int itr) {

    double prediction, error;
    double up_prediction, up_error, down_prediction, down_error;

    for (int i=0; i<itr; i++) {
        prediction = _input * _weight;
        error = powf((prediction - expected_value), 2);
        printf("Error : %f Prediction : %f \r\n", error, prediction);

        up_prediction = _input * (_weight + step_amount);
        up_error = powf((expected_value - up_prediction), 2);

        down_prediction = _input * (_weight - step_amount);
        down_error = powf((expected_value - down_prediction), 2);

        if (down_error < up_error) {
          _weight = _weight - step_amount;
        }
        if (down_error > up_error) {
            _weight = _weight + step_amount;
        }
    }
}

void normalize_data(double *input_vector, double *output_vector, int LEN) {

    int i;
    double max = input_vector[0];

    for (i=1; i<LEN; i++) {
        if (input_vector[i] > max) {
            max = input_vector[i];
        }
    }

    /* Normalize */
    for (i=0; i<LEN;i++) {
        output_vector[i] = input_vector[i] / max;
    }
}

void weight_random_init(int HIDDEN_LEN,
                        int INPUT_LEN,
                        double weights_matrix[HIDDEN_LEN][INPUT_LEN]) {
    double d_rand;

    srand(2);
    for (int i=0;i<HIDDEN_LEN;i++) {
        for (int j=0;j<INPUT_LEN;j++) {
            d_rand = (rand()%10);
            d_rand /= 10;

            weights_matrix[i][j] = d_rand;
        }
    }
}

void normalize_data_2d(int ROW, int COL, double input_matrix[ROW][COL], double output_matrix[ROW][COL]) {

    /* Find max number in matrix */
    double max = 0;
    for (int i=0; i<ROW;i++) {
        for (int j=0;j<COL;j++) {
            if (input_matrix[i][j] > max) {
                max = input_matrix[i][j];
            }
        }
    }
    for (int i=0;i<ROW;i++) {
        for (int j=0;j<COL;j++) {
            output_matrix[i][j] = input_matrix[i][j]/max;
        }
    }
}

void weight_random_init_1d(double *output_vector, int LEN) {

    double d_rand;
    srand(2);
    for (int i=0;i<LEN;i++) {
        d_rand = (rand()%10);
        d_rand /=  10;
        output_vector[i] = d_rand;
    }
}

double sigmoid(double x) {

    double result = 1/ (1 + exp(-x));
    return result;
}

void vector_sigmoid(double *input_vector, double *output_vector, int LEN) {

    for (int i=0;i<LEN;i++) {
        output_vector[i] = sigmoid(input_vector[i]);
    }
}
