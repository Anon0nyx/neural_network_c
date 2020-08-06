#include "nn_library.c"

int main() {

    const int num_inputs = 256;
    const int num_outputs = 10;

    float rate = 1.0f;
    const float eta = 0.99f;

    const int num_hid_layers = 28;
    const int iterations = 128;

    const Data data = build("semeion.data", num_inputs, num_outputs);

    const NeuralNetwork_Type network = NNbuild(num_inputs, num_hid_layers, num_outputs);

    for (int i = 0; i < iterations; i++) {

        shuffle(data);
        float error = 0.0f;
        for (int j = 0; j < data.rows; j++) {

            const float *const input = data.input[j];
            const float *const target = data.target[j];
            error += NNtrain(network, input , target, rate);
        }

        printf("Error %.12f :: Learning Rate %f :: Epoc %d\n", (double)error / data.rows, (double)rate, (int)i+1);
        rate *= eta;
    }

    NNsave(network, "mymodel.network");
    NNfree(network);

    const NeuralNetwork_Type my_loaded_model = NNload("mymodel.network");

    const float *const input = data.input[0];
    const float *const target = data.target[0];

    const float *const pd = NNpredict(my_loaded_model, input);

    NNprint(target, data.num_outputs);
    NNprint(pd, data.num_outputs);
    NNfree(my_loaded_model);
    dfree(data);

    return 0;
}
