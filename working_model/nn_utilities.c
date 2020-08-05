#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

typedef struct{

    float **in; /*2D array for inputs */
    float **tg; /*2D array for targets*/
    int nips;    /*Number of inputs    */
    int nops;    /*Number of outputs   */
    int rows;    /*Number of rows      */

}Data;


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

    for(int row=0; row < d.rows; row++){
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

void shuffle(const Data d){

    for(int a=0; a < d.rows; a++){
        const int b =  rand() %d.rows;
        float * ot = d.tg[a];
        float * it = d.in[a];

        d.tg[a] = d.tg[b];
        d.tg[b] = ot;

        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}
