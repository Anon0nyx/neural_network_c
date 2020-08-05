#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

typedef struct{

    float ** in; /*2D array for inputs */
    float ** tg; /*2D array for targets*/
    int nips;    /*Number of inputs    */
    int nops;    /*Number of outputs   */
    int rows;    /*Number of rows      */

}Data;

typedef struct
{
    float *w; /*All weights*/
    float *x; /*hidden layer to output layer weights*/
    float *b; /*biases*/
    float *h; /*hidden layer*/
    float *o; /*output layer*/
    int nb;   /*number of biases*/
    int nw;   /*number of weights*/
    int nips; /*number of inputs*/
    int nhid; /*number of hidden neurons*/
    int nops; /*number of outputs*/

}NeuralNetwork_Type;

static float toterr(const float * const tg,const float * const o, const int size);

// One line functions
static float err(const float a, const float b) { return 0.5f*(a-b)*(a-b); }
static float pderr(const float a, const float b) { return a - b; }
static float act(const float a) { return 1.0f/(1.0f + expf(-a)); }
static float pdact(const float a ) { return a*(1.0f -a); }
static float frand() { return rand()/(float)RAND_MAX; }


static void fprop(const NeuralNetwork_Type nn,const float * const in){

 /*Hidden layer neuron values*/
   for(int i=0;i<nn.nhid;i++){
      float sum = 0.0f;
          for(int j=0;j<nn.nips;j++){
              sum +=in[j]*nn.w[i*nn.nips+j];
          }
          nn.h[i] =  act(sum + nn.b[0]);
      }

      /*Output layer neuron values*/
      for(int i=0;i<nn.nops;i++){
          float sum  =0.0f;
          for(int j=0 ;j<nn.nhid;j++){
              sum += nn.h[j] * nn.x[i*nn.nhid+j];
          }
          nn.o[i] = act(sum+nn.b[1]);
      }
}

static void bprop(const NeuralNetwork_Type nn,
                  const float *const in,
                  const float * const tg,
                  float rate) {
    for(int i=0;i<nn.nhid;i++){
        float sum =0.0f;
        for(int j=0;j<nn.nops;j++){
            const float a =  pderr(nn.o[j],tg[j]);
            const float b = pdact(nn.o[j]);

            sum += a * b * nn.x[j*nn.nhid + i];

            nn.x[j * nn.nhid + i ] -= rate *a*b*nn.h[i];
          }
          for(int j=0;j<nn.nips;j++){
              nn.w[i * nn.nips + j] -= rate *sum *pdact(nn.h[i])*in[j];
          }
      }
}

static  void wbrand(const NeuralNetwork_Type nn) {
     for(int i =0;i<nn.nw;i++){
         nn.w[i] = frand() -0.5f;
     }
     for(int i=0;i<nn.nb;i++){
         nn.b[i] = frand() - 0.5f;
     }
}

float * NNpredict(const NeuralNetwork_Type nn, const float * in ) {

   fprop(nn,in);
   return nn.o;
}


NeuralNetwork_Type NNbuild(const int nips,const int nhid,const int nops){

     NeuralNetwork_Type nn;
     nn.nb = 2;                                       /*number of biases*/
     nn.nw =  nhid * (nips +nops);                  /*number of weights*/
     nn.w =  (float *)calloc(nn.nw,sizeof(*nn.w));  /*All weights*/
     nn.x =  nn.w + nhid *nips;                     /*hidden layer to output layer weights*/
     nn.b = (float *)calloc(nn.nb,sizeof(*nn.b));   /*biases*/
     nn.h = (float *)calloc(nhid,sizeof(*nn.h));   /*hidden layer*/
     nn.o = (float *)calloc(nops,sizeof(*nn.o));    /*output layer*/
     nn.nips = nips;                              /*number of inputs*/
     nn.nhid =  nhid;                             /*number of hidden neurons*/
     nn.nops = nops;                              /*number of outputs*/
     wbrand(nn);

     return nn;
}

void NNsave(const NeuralNetwork_Type nn, const char * path) {

    FILE * const file =  fopen(path,"w");
    /*Save the header*/
    fprintf(file,"%d %d %d\n",nn.nips,nn.nhid,nn.nops);

    /*Save the biases*/
    for(int i =0;i<nn.nb;i++){
        fprintf(file,"%f\n",(double)nn.b[i]);
    }

    /*Save the weights*/
    for(int i=0;i<nn.nw;i++){
        fprintf(file,"%f\n",(double)nn.w[i]);
    }
    fclose(file);
}

NeuralNetwork_Type NNload(const char * path){

    FILE * const file = fopen(path,"r");
    int nips =0;
    int nhid = 0;
    int nops =0;

    /*Load the header*/
    fscanf(file, "%d %d %d\n",&nips,&nhid,&nops);

    const NeuralNetwork_Type nn =  NNbuild(nips,nhid,nops);

    /*Load the biases*/
    for(int i=0;i<nn.nb;i++){
        fscanf(file,"%f\n",&nn.b[i]);
    }
    /*Load the weights*/
    for(int i =0;i<nn.nw;i++){
        fscanf(file,"%f\n",&nn.w[i]);
    }
    fclose(file);
    return nn;
}


float NNtrain(const NeuralNetwork_Type nn, const float * in,const float * tg,float rate){

     fprop(nn,in);
     bprop(nn,in,tg,rate);

     return toterr(tg,nn.o,nn.nops);

}

void NNprint(const float * arr, const int size){

    double max =  0.0f;
    int idx;

    for(int i=0;i <size;i++){

        printf("%f ",(double)arr[i]);

        if(arr[i] > max){
            idx = i;
            max =  arr[i];
        }
    }

    printf("\n");
    printf("The number is :%d\n",idx);
}

void NNfree(const NeuralNetwork_Type nn){

    free(nn.w);
    free(nn.b);
    free(nn.h);
    free(nn.o);
}


static float toterr(const float * const tg,const float * const o, const int size) {

    float sum= 0.0f;
    for(int i=0;i<size;i++){
        sum +=err(tg[i],o[i]);
    }

    return sum;
}

int lns(FILE *const file){

  int ch =  EOF;
  int lines = 0;
  int pc = '\n';

  while((ch = getc(file))!=EOF){

    if(ch == '\n'){
        lines++;
    }
    pc = ch;
  }

   if(pc !='\n'){
    lines++;
   }
   rewind(file);
   return lines;
}

char * readln(FILE * const file){

  int ch =  EOF;
  int reads = 0;
  int size = 128;
  char * line =  (char *)malloc((size)*sizeof(char));

  while((ch =  getc(file)) != '\n' && ch !=EOF){

    line[reads++] =  ch;
    if(reads +1 == size){
        line = (char *)realloc((line),(size *=2)*sizeof(char));
    }
  }

  line[reads] ='\0';
  return line;


}


float ** new2d(const int rows, const int cols){

    float **row = (float**)malloc((rows)*sizeof(float *));

    for(int r=0;r<rows;r++){
        row[r] = (float *)malloc((cols)*sizeof(float));
    }
    return row;
}

Data ndata(const int nips,const int nops, const int rows){

  const Data data = {
        new2d(rows,nips),
        new2d(rows,nops),
        nips,
        nops,
        rows
  };

  return data;

}

void parse(const Data data,char * line,const int row){

 const int cols = data.nips + data.nops;

 for(int col =0;col<cols;col++){

      const float val  = atof(strtok(col == 0 ? line :NULL," "));

     if(col < data.nips)
        data.in[row][col] = val;
     else
        data.tg[row][col - data.nips] = val;


 }



}

void dfree(const Data d){


    for(int row =0;row<d.rows;row++){
        free(d.in[row]);
        free(d.tg[row]);
    }

    free(d.in);
    free(d.tg);

}


void shuffle(const Data d){

    for(int a =0;a<d.rows;a++){

        const int b =  rand() %d.rows;
        float * ot = d.tg[a];
        float * it = d.in[a];


        d.tg[a] = d.tg[b];
        d.tg[b] = ot;

        d.in[a] = d.in[b];
        d.in[b] = it;
    }



}

Data build(const char * path,const int nips,const int nops){

    FILE * file = fopen(path,"r");

    if(file ==NULL){

    printf("Could not open %s\n",path);
    printf("Dataset does not exist \n");
    exit(1);
    }


    const int rows  =lns(file);
    Data data =  ndata(nips,nops,rows);

    for(int row =0;row<rows;row++){
        char * line = readln(file);
        parse(data,line,row);
        free(line);
    }

    fclose(file);
    return data;

}

int main () {
   const int nips = 256;
   const int nops = 10;

   float rate = 1.0f;
   const float eta = 0.99f;

   const int nhid = 28;
   const int iterations =  128;


   const Data data =  build("semeion.data",nips,nops);

   const NeuralNetwork_Type nn = NNbuild(nips,nhid,nops);

   for(int i=0;i<iterations;i++){

    shuffle(data);
    float error = 0.0f;
    for(int j=0;j<data.rows;j++){

        const float * const in = data.in[j];
        const float * const tg = data.tg[j];
        error +=NNtrain(nn,in,tg,rate);
    }

    printf("Error %.12f :: learning rate %f\n",(double)error/data.rows,(double)rate);
    rate *=eta;
   }

   NNsave(nn,"mymodel.nn");
   NNfree(nn);

   const NeuralNetwork_Type my_loaded_model = NNload("mymodel.nn");

   const float * const in =  data.in[0];
   const float * const tg =  data.tg[0];

   const float * const pd = NNpredict(my_loaded_model,in);

   NNprint(tg,data.nops);
   NNprint(pd,data.nops);
   NNfree(my_loaded_model);
   dfree(data);

   return 0;
}
