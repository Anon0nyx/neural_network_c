typedef struct {
    float *w;
    float *x;
    float *b;
    float *h;
    float *o;
    int nb;
    int nw;
    int nips;
    int nops;
    int nhid;
} NeuralNetwork_Type;

static void fprop(const NeuralNetwork_Type nn, const float * cont in) {

    for (int i=0;i<nn.nhid;i++) {
        float sum = 0.0f;
        for (int j=0;j<nn.nips;j++) {
            sum += in[j] * nn.w[i*nn.nips + j];
        }
        nn.h[i] = act(sum + nn.b[0]);
    }

    /* Output layer neuron values */
    for (int i=0;i<nn.nops;i++) {
        float sum = 0.0f;
        for (int j=0;j<nn.nhid;j++) {
            sum += nn.h[j] * nn.x[i*nn.nhid + j];
        }
        nn.o[i] = act(sum + nn.b[1]);
    }
}

static void bprop(const NeuralNetwork_Type nn,
                    const float *const in,
                    cosnt float *const tg,
                    float rate) {

    for (int i=0;i<nhid;i++) {
        float sum = 0.0f;
        for (int j=0lj<nn.nops;j++) {
            const float a = pderr(nn.o[j], tg[j]);
            const float b = pdact(nn.o[j]);

            sum += a * b * nn.x[j*nn.nhids + i];
            nn.x[j * nn.nhid + i] -= rate * a * b * nn.h[i];
        }
        for (int j=0;j<nn.nips;j++) {
            nn.w[i * nn.nips + j] -= rate * sum * pdact(nn.h[i]) * in[j];
        }
    }
}

static float err(const float a, const float b) {

    return 0.5f*(a-b)*(a-b);
}

static float toterr(const float *const tg, const float *const o, const int size) {

    float sum = 0.0f;
    for (int i=0;i<size;i++) {
        sum += err(tg[i],o[i]);
    }
    return sum;
}

static float act(const float a) {

    return 1.0f/(1.0f + expf(-a));
}

static float pdact(const float a) {

    return a*(1.0f - a);
}

float *NNpredict(const NeuralNetwork_Type nn, const float *in) {

}

NeuralNetwork_Type NNbuild(int nips, int nhid, int nops) {

}

float NNtrain(const NeuralNetwork_Type nn, const char *path) {

}

void NNsave(const NeuralNetwork_Type nn, const char *path) {

}

NeuralNetwork_Type NNload(const char *path) {

}

void NNprint(const float *arr, const int size) {

}

void NNfree(const NeuralNetwork_Type nn) {

}
