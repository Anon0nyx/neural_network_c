#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

typedef struct {
    float **input;      /*2D array for inputs */
    float **target;     /*2D array for targets*/
    int num_inputs;     /*Number of inputs    */
    int num_outputs;    /*Number of outputs   */
    int rows;           /*Number of rows      */
}Data;


int lns(FILE *const file) {

    int ch =  EOF;
    int lines = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF) {
        if(ch == '\n') {
            lines++;
        }
        pc = ch;
    }
    if(pc != '\n') {
        lines++;
    }
    rewind(file);
    return lines;
}

char *readln(FILE *const file) {
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char *line = (char *)malloc((size)*sizeof(char));
    while((ch = getc(file)) != '\n' && ch != EOF) {
        line[reads++] = ch;
        if(reads+1 == size) {
            line = (char *)realloc((line), (size*=2) * sizeof(char));
        }
    }
    line[reads] = '\0';
    return line;
}

float **new2d(const int rows, const int cols) {

    float **row = (float**)malloc((rows) * sizeof(float *));
    for(int r=0; r < rows; r++){
        row[r] = (float *)malloc((cols) * sizeof(float));
    }
    return row;
}

Data ndata(const int num_inputs, const int num_outputs, const int rows) {

    const Data data = {
        new2d(rows, num_inputs),
        new2d(rows, num_outputs),
        num_inputs,
        num_outputs,
        rows
    };
    return data;
}

void parse(const Data data, char *line, const int row) {

    const int cols = data.num_inputs + data.num_outputs;
    for(int col=0; col < cols; col++) {
        const float val = atof(strtok(col == 0 ? line :NULL, " "));
        if(col < data.num_inputs)
            data.input[row][col] = val;
        else
            data.target[row][col - data.num_inputs] = val;
    }
}

void dfree(const Data data) {

    for(int row=0; row < data.rows; row++) {
        free(data.input[row]);
        free(data.target[row]);
    }
    free(data.input);
    free(data.target);
}

void shuffle(const Data data) {

    for(int a=0; a < data.rows; a++) {
        const int b =  rand() % data.rows;
        float *ot = data.target[a];
        float *it = data.input[a];

        data.target[a] = data.target[b];
        data.target[b] = ot;

        data.input[a] = data.input[b];
        data.input[b] = it;
    }
}
