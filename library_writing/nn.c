#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static float act(const float a);
static float pdact(const float a);
static float err(const float a, const float b);
static float toterr(const float * const tg, const float * const o, const int size);
static float pderr(const float a, const float b);
static float frand();

typedef struct NeuralNetwork_Type NeuralNetwork_Type;

struct NeuralNetwork_Type{
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
};

static void fprop(const NeuralNetwork_Type nn, const float *const in) {

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
                    const float *const tg,
                    float rate) {

    for (int i=0;i<nn.nhid;i++) {
        float sum = 0.0f;
        for (int j=0;j<nn.nops;j++) {
            const float a = pderr(nn.o[j], tg[j]);
            const float b = pdact(nn.o[j]);

            sum += a * b * nn.x[j*nn.nhid + i];
            nn.x[j * nn.nhid + i] -= rate * a * b * nn.h[i];
        }
        for (int j=0;j<nn.nips;j++) {
            nn.w[i * nn.nips + j] -= rate * sum * pdact(nn.h[i]) * in[j];
        }
    }
}

static void wbrand(const NeuralNetwork_Type nn) {

    for (int i=0;i<nn.nw;i++) {
        nn.w[i] = frand() - 0.5f;
    }
    for (int i=0;i<nn.nb;i++) {
        nn.b[i] = frand() - 0.5f;
    }
}

float *NNpredict(const NeuralNetwork_Type nn, const float * in) {

    fprop(nn, in);
    return nn.o;
}

NeuralNetwork_Type NNbuild(int nips, int nhid, int nops) {

    NeuralNetwork_Type nn;
    nn.nb = 2;
    nn.nw = nhid * (nips + nops);
    nn.w = (float *)calloc(nn.w,sizeof(*nn.w));
    nn.x = nn.w + nhid *nips;
    nn.b = (float *)calloc(nn.nb, sizeof(*nn.b));
    nn.h = (float *)calloc(nhid,sizeof(*nn.b));
    nn.o = (float *)calloc(nops,sizeof(*nn.o));
    nn.nips = nips;
    nn.nhid = nhid;
    nn.nops = nops;
    wbrand(nn);

    return nn;
}

float NNtrain(const NeuralNetwork_Type nn, const float * in, const float * tg, float rate) {

    fprop(nn,in);
    bprop(nn,in,tg,rate);

    return toterr(tg,nn.o,nn.nops);
}

void NNsave(const NeuralNetwork_Type nn, const char * path) {

    FILE * const file = fopen(path,"w");
    fprintf(file,"%d %d %d\n",nn.nips,nn.nhid,nn.nops);
    for (int i=0;i<nn.nb;i++) {
        fprintf(file,"%f\n", (double)nn.b[i]);
    }
    for (int i=0;i<nn.nw;i++) {
        fprintf(file,"%f\n", (double)nn.w[i]);
    }
}

NeuralNetwork_Type NNload(const char * path) {

    FILE * const file = fopen(path,"r");
    int nips = 0;
    int nhid = 0;
    int nops = 0;

    fscanf(file, "%d %d %d\n", &nips, &nhid, &nops);
    const NeuralNetwork_Type nn = NNbuild(nips, nhid, nops);

    for (int i=0;i<nn.nb;i++) {
        scanf(file,"%f\n",&nn.b[i]);
    }
    for (int i=0;i<nn.nw;i++) {
        scanf(file,"%f\n",&nn.w[i]);
    }

    fclose(file);
    return nn;
}

void NNprint(const float * arr, const int size) {

    double max = 0.0f;
    int idx;

    for (int i=0;i<size;i++) {
        printf("%f\t", (double)arr[i]);
        if (arr[i] > max) {
            idx = 1;
            max = arr[i];
        }
    }
    printf("\n\r");
    printf("The number is : %d\n", idx);
}

void NNfree(const NeuralNetwork_Type nn) {

    free(nn.w);
    free(nn.b);
    free(nn.h);
    free(nn.o);
}

static float err(const float a, const float b) {

    return 0.5f*(a-b)*(a-b);
}

static float pderr(const float a, const float b){

   return a - b;
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

static float frand() {

   return rand()/(float)RAND_MAX;
}
