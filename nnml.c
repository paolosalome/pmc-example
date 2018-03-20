/*******************************************************************************
*   original code by                                                           *
*   JOHN BULLINARIA  2004. Modified by Massimo Bernaschi 2016                  *
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>

#include "common.h"
#include "dictionary.h"
#include "iniparser.h"

#define REAL float
#define NULLFILE "/dev/null"
#define DEFMAXEPOCH 1000
#if !defined(MAX)
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

void Usage(char *cmd) {
    printf("-----------------------\n");
    printf("Neural Networks Learning Code (by backpropagation)\n");
    printf("-----------------------\n");
    printf("Usage: %s \n"
           "-i inputfile  \n"
           "[-v verbose \n -D Debug \n  -h ThisHelp]\n",
           cmd);
}

void ReadFromFile(char *fn,void **array,int nrow, int ncol, int ts) {
  FILE *fp=NULL;
  int i, j;
  double **dp;
  float  **sp;
  switch(ts) {
      case sizeof(float):
        sp=(float **)array;
        break;
      case sizeof(double):
        dp=(double **)array;
        break;
      default:
        writelog(TRUE,APPLICATION_RC,"invalid size in ReadFromFile: %d\n",ts);
        break;
  }
  fp=Fopen(fn,"r");
  for(i=0; i<nrow; i++) {
    for(j=0; j<ncol; j++) {
      switch(ts) {
      case sizeof(float):
        fscanf(fp,"%f",&(sp[i][j]));
        break;
      case sizeof(double):
        fscanf(fp,"%lf",&(dp[i][j]));
        break;
      default:
        writelog(TRUE,APPLICATION_RC,"invalid size in ReadFromFile: %d\n",ts);
        break;
      }
    }
  }
  fclose(fp);
}

void ReadFromFileArr(char *fn,void *array,int nrow, int ncol, int ts) {
  FILE *fp=NULL;
  int i, j;
  double **dp;
  float  **sp;
  switch(ts) {
      case sizeof(float):
        sp=(float *)array;
        break;
      case sizeof(double):
        dp=(double *)array;
        break;
      default:
        writelog(TRUE,APPLICATION_RC,"invalid size in ReadFromFile: %d\n",ts);
        break;
  }
  fp=Fopen(fn,"r");
  for(i=0; i<nrow; i++) {
    for(j=0; j<ncol; j++) {
      switch(ts) {
      case sizeof(float):
        fscanf(fp,"%f",&(sp[i*ncols+ j]));
        break;
      case sizeof(double):
        fscanf(fp,"%lf",&(dp[i*ncols+ j]));
        break;
      default:
        writelog(TRUE,APPLICATION_RC,"invalid size in ReadFromFile: %d\n",ts);
        break;
      }
    }
  }
  fclose(fp);
}

void HOST_init_struct( DATA* WeightH2H, DATA* BiasH2H, DATA* DeltaWeightH2H, DATA* DeltaBiasH2H, DATA* Delta, DATA* H2H,	int* matrix_W_index, int* matrix_B_index,  int* matrix_DELTA_index, int* matrix_H2H_index, int* nupl, int layers, int NumPattern, int smallwt) {
		
	int prev_sum[4];
	matrix_H2H_index[0] = 0;
	matrix_DELTA_index[0] = 0;
	matrix_W_index[0] = 0;
	matrix_B_index[0] = 0;
	//Bisogner� inserire i controlli sulle malloc
	/*il padding della matrice al layer corrente dipende da quello dei layer precedenti*/

	for (int layer = 1; layer<(layers - 1); layer++) {

		prev_sum[0] = matrix_H2H_index[layer - 1];
		prev_sum[1] = matrix_DELTA_index[layer - 1];
		prev_sum[2] = matrix_W_index[layer - 1];
		prev_sum[3] = matrix_B_index[layer - 1];

		matrix_H2H_index[layer] = nupl[layer - 1] * NumPattern + prev_sum[0];
		matrix_DELTA_index[layer] = nupl[layer] * NumPattern + prev_sum[1];
		matrix_W_index[layer] = nupl[layer - 1] * nupl[layer] + prev_sum[2];
		matrix_B_index[layer] = nupl[layer] + prev_sum[3];

		for (int j = 0; j < nupl[layer + 1]; j++) {
      BiasH2H[matrix_B_index[layer] + j] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
			DeltaBiasH2H[matrix_B_index[layer] + j] = 0.0f;
      for (int i = 0; i < nupl[layer]; i++) {
				WeightH2H[matrix_W_index[layer] + i*nupl[layer + 1] + j] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
				DeltaWeightH2H[matrix_W_index[layer] + i*nupl[layer + 1] + j] = 0.0f;
			}
		}
    
	}
	prev_sum[0] = matrix_H2H_index[layers - 2];
	matrix_H2H_index[layers - 1] = nupl[layers - 2] * NumPattern + prev_sum[0];

	for (int j = 0; j < nupl[1]; j++) {
    BiasH2H[j] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
  	DeltaBiasH2H[j] = 0.0f;
    for (int i = 0; i < nupl[0]; i++) {
			WeightH2H[i*nupl[1] + j] = 2.0 * ( drand48() - 0.5 ) * smallwt ;
			DeltaWeightH2H[i*nupl[1] + j] = 0.0f;
    }
  }
}


int main(int argc, char *argv[]) {
    int     h, i, j, k, p, np, op, epoch;
    int    NumPattern, NumInput, NumHidden, NumOutput;
/*
    double Input[NUMPAT+1][NUMIN+1] = { 0, 0, 0,  0, 0, 0,  0, 1, 0,  0, 0, 1,  0, 1, 1 };
    double Target[NUMPAT+1][NUMOUT+1] = { 0, 0,  0, 0,  0, 1,  0, 1,  0, 0 };
    double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
    double DeltaO[NUMPAT+1][NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMPAT+1][NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
    double Error, eta = 0.5, alpha = 0.9, smallwt = 0.5;
*/
    DATA* WeightH2H, *BiasH2H, *DeltaWeightH2H, *DeltaBiasH2H, *Delta, *H2H, *TempDeltaWeightH2H, *TempDeltaBiasH2H;
    DATA* H_WeightH2H,*H_BiasH2H,*H_DeltaWeightH2H,*H_DeltaBiasH2H,*H_Delta,*H_H2H;
    

    DATA *INPUT_MAT, *ERROR_MAT, *DEV_ERROR_MAT;
    DATA *ERROR, *DEV_ERROR;
    DATA *TARGET;
    cudaStream_t streams[NSTREAMS];


    REAL **Input;
    REAL **Target;
    REAL **Sum;
    REAL **WeightIH, **Hidden;
    REAL ***H2H, ***DeltaH2H;
    REAL ***WeightH2H, ***DeltaWeightH2H;
    REAL **Htemp, **Deltatemp;
    REAL **WeightHO, **Output;
    REAL **DeltaO, **DeltaH;
    REAL **DeltaWeightIH, **DeltaWeightHO;
    REAL Error, Eps, eta, alpha, smallwt;
    int *ranpat;
    int verbose=FALSE;
    int maxepoch=DEFMAXEPOCH;
    int NumHL=1;
    int *nupl=NULL;
    char *inputfile = NULL;
    char *po;
    dictionary *ini;
    char key[MAXSTRLEN];
    char formatstring[MAXSTRLEN];
    char LogFileName[MAXSTRLEN];
    char InputFileName[MAXSTRLEN];
    char TargetFileName[MAXSTRLEN];
    char ResultFileName[MAXSTRLEN];
    char RestartFileName[MAXSTRLEN];
    char DeltaFileName[MAXSTRLEN];
    char RestartDeltaFileName[MAXSTRLEN];
    FILE *fp=NULL;
    FILE *fpd=NULL;
    FILE *fpl=NULL;
    if(sizeof(REAL)==sizeof(float)) {
        strcpy(formatstring,"%f ");
    }
    if(sizeof(REAL)==sizeof(double)) {
        strcpy(formatstring,"%lf ");
    }

    for(i = 1; i < argc; i++) {
      po = argv[i];
      if (*po++ == '-') {
        switch (*po++) {
        case 'h':
          Usage(argv[0]);
          exit(OK);
          break;
        case 'v':
          verbose=TRUE;
          break;
        case 'i':
          SKIPBLANK
          inputfile=Strdup(po);
          break;
        default:
          Usage(argv[0]);
          exit(OK);
          break;
        }
      }
    }
    if(inputfile==NULL) {
      Usage(argv[0]);
      exit(OK);
    }

    ini = iniparser_load(inputfile);

    if(ini==NULL) { writelog(TRUE,APPLICATION_RC,"Cannot parse file: %s\n", inputfile); }

    READINTFI(maxepoch,"Max number of epochs");
    READINTFI(NumPattern,"Number of training data");
    READINTFI(NumInput,"Number of input units");
    READINTFI(NumOutput,"Number of output units");
    READINTFI(NumHL,"Number of hidden layers");
    READREALFI(eta,"Learning rate");
    READREALFI(alpha,"Momentum");
    READREALFI(smallwt,"Initialization scale");
    READREALFI(Eps,"Error threshold");
    {READSTRFI(LogFileName,"Log file name");}
    {READSTRFI(InputFileName,"Input file name");}
    {READSTRFI(TargetFileName,"Target file name");}
    {READSTRFI(ResultFileName,"Results file name");}
    {READSTRFI(DeltaFileName,"Result delta file name");}
    {READSTRFI(RestartFileName,"Restart file name");}
    {READSTRFI(RestartDeltaFileName,"Restart delta file name");}
    nupl=makevect(NumHL+2,sizeof(int));
    nupl[0]=NumInput;
    nupl[NumHL+1]=NumOutput;
    if(NumHL) {
      int scratch;
      char tempstring[MAXSTRLEN];
      for(i=1; i<=NumHL; i++) {
        snprintf(tempstring,sizeof(tempstring),"Number of units in layer %d",i-1);
        READINTFI(scratch,tempstring);
        nupl[i]=scratch;
      }
    }

    int TOTAL_LAYER = NumHL +2; //#of layers
    int NSTREAMS = 3;
    int STREAMSIZE = NumPattern/NSTREAMS;
		int GLOBAL_H_SIZE= NumPattern*NumInput, GLOBAL_DELTA_SIZE=0, GLOBAL_W_SIZE=0, GLOBAL_BIAS_SIZE=0;

    for(int numl=1;numl<TOTAL_LAYER;numl++){
      GLOBAL_H_SIZE += nupl[numl]*NumPattern;
      GLOBAL_DELTA_SIZE += nupl[numl]*NumPattern;
      GLOBAL_W_SIZE += nupl[numl-1]*nupl[numl];
      GLOBAL_BIAS_SIZE += nupl[numl];
    }
/*++++------------------------------------ERRORS--------------------------------------------------++++*/

    ERROR_MAT = (DATA*)malloc(NumPattern*NumOutput * sizeof(DATA)); // ERROR FOR CHECKING CORRECTNESS
    HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR_MAT, NumPattern*NumOutput * sizeof(DATA))); //DEVICE ERROR MAT

    ERROR = (DATA*)malloc(sizeof(DATA)); // ERROR FOR CHECKING CORRECTNESS
    HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR, sizeof(DATA))); //DEVICE ERROR
    HANDLE_CUDA(cudaMemset(DEV_ERROR, 0, sizeof(DATA)));

    /*----------------------------------------ERRORS END--------------------------------------------------*/
  /*++++---------------------------init INPUT_MAT and TARGET (HOST)-----------------------------++++*/
    
    TARGET = (DATA*)malloc(NumOutput*NumPattern * sizeof(DATA)); //TARGET OF THE PATTERNS
    /*INPUT_MAT is pinned memory*/  
    HANDLE_CUDA(cudaHostAlloc(&INPUT_MAT, NumInput * NumPattern * sizeof(DATA), 0));

    ReadFromFileArr(InputFileName,INPUT_MAT,NumPattern,NumInput,sizeof(REAL));
    ReadFromFileArr(TargetFileName,TARGET,NumPattern,NumOutput,sizeof(REAL));

	/*---------------------------end init INPUT_MAT and TARGET (HOST)-------------------------*/

  /*++++---------------------------data structures on host and device-------------------------++++*/
	/*	HOST MATRIX AND INDEXES FOR PADDING */
    //H_WeightH2H = (DATA*)malloc(GLOBAL_W_SIZE*sizeof(DATA));
    H_WeightH2H = (DATA*)makevect(GLOBAL_W_SIZE,sizeof(DATA));
    //H_BiasH2H = (DATA*)malloc(GLOBAL_BIAS_SIZE*sizeof(DATA));
    H_BiasH2H = (DATA*)makevect(GLOBAL_BIAS_SIZE,sizeof(DATA));
    //H_DeltaWeightH2H = (DATA*)calloc(GLOBAL_W_SIZE,sizeof(DATA));
    H_DeltaWeightH2H = (DATA*)makevect(GLOBAL_W_SIZE,sizeof(DATA));
    //H_DeltaBiasH2H = (DATA*)calloc(GLOBAL_BIAS_SIZE*sizeof(DATA));
    H_DeltaBiasH2H = (DATA*)makevect(GLOBAL_BIAS_SIZE,sizeof(DATA));
    //H_Delta = (DATA*)malloc(GLOBAL_DELTA_SIZE*sizeof(DATA));
    H_Delta = (DATA*)makevect(GLOBAL_DELTA_SIZE,sizeof(DATA));
    //H_H2H = (DATA*)malloc(GLOBAL_H_SIZE*sizeof(DATA));
    H_H2H = (DATA*)makevect(GLOBAL_H_SIZE,sizeof(DATA));

    int H_matrix_W_index[TOTAL_LAYER - 1];//INDEX for padding in Weight 
    int H_matrix_B_index[TOTAL_LAYER - 1];//INDEX for padding in Bias 
    int H_matrix_DELTA_index[TOTAL_LAYER - 1];//INDEX for padding in Delta
    int H_matrix_H2H_index[TOTAL_LAYER];//INDEX for padding in H2H

    //Init weights and biases on host
    HOST_init_struct(H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_Delta, H_H2H,	H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index, nupl, TOTAL_LAYER, NumPattern, smallwt);
    //    DEVICE    Malloc the necessary space on device memory
    HANDLE_CUDA(cudaMalloc((void **)&WeightH2H , GLOBAL_W_SIZE * sizeof(DATA)));
    HANDLE_CUDA(cudaMalloc((void **)&BiasH2H, GLOBAL_BIAS_SIZE * sizeof(DATA)));
    HANDLE_CUDA(cudaMalloc((void **)&DeltaWeightH2H, GLOBAL_W_SIZE * sizeof(DATA)));
    HANDLE_CUDA(cudaMalloc((void **)&DeltaBiasH2H, GLOBAL_BIAS_SIZE * sizeof(DATA)));
    HANDLE_CUDA(cudaMalloc((void **)&Delta, GLOBAL_DELTA_SIZE * sizeof(DATA)));
    HANDLE_CUDA(cudaMalloc((void **)&H2H, GLOBAL_H_SIZE * sizeof(DATA)));
    HANDLE_CUDA(cudaMalloc((void **)&TempDeltaWeightH2H, NSTREAMS*GLOBAL_W_SIZE *sizeof(DATA)));
    HANDLE_CUDA(cudaMalloc((void **)&TempDeltaBiasH2H, NSTREAMS*GLOBAL_BIAS_SIZE *sizeof(DATA)));

	/*---------------------------end data structures on host and device----------------------------*/
  /*++++---------------------------cuda array for texture-----------------------------++++*/
    cudaArray* DEV_TARGET_CUDA;
    cudaChannelFormatDesc channel;

    channel = cudaCreateChannelDesc<DATA>();
    HANDLE_CUDA(cudaMallocArray(&DEV_TARGET_CUDA, &channel, NumOutput, NumPattern));
    HANDLE_CUDA(cudaMemcpyToArray(DEV_TARGET_CUDA, 0, 0, TARGET, NumOutput*NumPattern * sizeof(DATA), cudaMemcpyHostToDevice));

    texreference_target.filterMode = cudaFilterModePoint; //turn off the interpolation of cudaFilterModeLinear
    texreference_target.addressMode[0] = cudaAddressModeWrap;//works in normalized coordinates only
    texreference_target.addressMode[1] = cudaAddressModeClamp;//works in both unnormalized and normalized coordinates

    HANDLE_CUDA(cudaBindTextureToArray(texreference_target, DEV_TARGET_CUDA)); //Texture reference binding
    /*---------------------------end cuda array for texture-------------------------*/

    /*++++-----------Streams creation------------++++*/
    for (int i = 0; i < NSTREAMS; i++) {
      HANDLE_CUDA(cudaStreamCreate(&streams[i]));
    }
  /*---------------end--streams creation-----------*/

    if(strcmp(LogFileName,NULLFILE)) { fpl=Fopen(LogFileName,"w"); }

    if(strcmp(RestartFileName,NULLFILE)) {
        fp=Fopen(RestartFileName,"r");
        if(strcmp(RestartDeltaFileName,NULLFILE)) {
                fpd=Fopen(RestartDeltaFileName,"r");
        }
        for( k = 0 ; k < nupl[NumHL+1] ; k ++ ) {
                fscanf(fp,formatstring,&H_BiasH2H[H_matrix_B_index[TOTAL_LAYER-2]+k]);//WeightHO[0][k]);
                if(fpd) fscanf(fpd,formatstring,&DeltaBiasH2H[H_matrix_B_index[TOTAL_LAYER-2]+k]);//DeltaWeightHO[0][k]);
                for( j = 0 ; j < nupl[NumHL] ; j++ ) {
                    fscanf(fp,formatstring,&H_WeightH2H[H_matrix_W_index[TOTAL_LAYER-2] + j*nupl[NumHL+1]+k]);//WeightHO[j][k]);
                    if(fpd) fscanf(fpd,formatstring,&H_DeltaWeightH2H[H_matrix_W_index[TOTAL_LAYER-2] + j*nupl[NumHL+1]+k]);//DeltaWeightHO[j][k]);
                }
                fscanf(fp,"\n");
                if(fpd) fscanf(fpd,"\n");
        }

        for(h=NumHL; h>0; h--) {
              for( j = 0 ; j < nupl[h] ; j++ ) {
                fscanf(fp,formatstring,&H_BiasH2H[H_matrix_B_index[h-1]+j]);    //WeightH2H[h-1][0][j]);
                if(fpd) fscanf(fpd,formatstring,&H_DeltaBiasH2H[H_matrix_B_index[h-1]+j]); //DeltaWeightH2H[h-1][0][j]);
                for( i = 0 ; i < nupl[h-1] ; i++ ) {
                    fscanf(fp,formatstring,&H_WeightH2H[H_matrix_W_index[h-1]+i*nupl[h]+j]); //WeightH2H[h-1][i][j]);
                    if(fpd) fscanf(fpd,formatstring,&H_DeltaWeightH2H[H_matrix_W_index[h-1]+i*nupl[h]+j]);//DeltaWeightH2H[h-1][i][j]);
                }
                fscanf(fp,"\n");
                if(fpd)fscanf(fpd,"\n");
              }
        }

        if (fp) fclose(fp);
        if (fpd) fclose(fpd);
    }
    if(verbose) {
      printf("\nInitial Bias and Weights\n");
      for( k = 0 ; k <  nupl[NumHL+1] ; k ++ ) {
                printf("Bias H to O[%d]: %f\n",k, H_BiasH2H[H_matrix_B_index[TOTAL_LAYER-2]+k]);//WeightHO[0][k]);
                for( j = 0 ; j < nupl[NumHL] ; j++ ) {
                    printf("Weight H[%d] to O[%d]: %f\n",j,k, H_WeightH2H[H_matrix_W_index[TOTAL_LAYER-2] + j*nupl[NumHL+1]+k]);//WeightHO[j][k]);
                }
      }
      for(h=NumHL; h>0; h--) {
        for( j = 0 ; j < nupl[h] ; j++ ) {
                printf("Bias[%d][%d]: %f\n",h-1,j,H_BiasH2H[H_matrix_B_index[h-1]+j]);//WeightH2H[h-1][0][j]);
                for( i = 0 ; i < nupl[h-1] ; i++ ) {
                  printf("Weight[%d][%d][%d]: %f\n",h-1,i,j,H_WeightH2H[H_matrix_W_index[h-1]+i*nupl[h]+j]);//WeightH2H[h-1][i][j]);
                }
        }
      }
    }
    
    cudaEvent_t start, stop;
    for( epoch = 0 ; epoch < maxepoch ; epoch++) {    /* iterate weight updates */

     /*++++-----------------------------------FEEDFORWARD-------------------------------------------++++*/

    
      printf("\nEPOCH %d\n",epoch+1);
      startTimer(&start, &stop);
      //feedforward(INPUT_MAT, htdm, dev_htdm, DEV_ERROR_MAT, DEV_ERROR, nupl, TOTAL_LAYER, streams, 1);
      BOOL first=(epoch==0)?1:0;
      feedforward(INPUT_MAT,H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index, /*HOST*/
        WeightH2H, BiasH2H, Delta, H2H, DeltaWeightH2H, DeltaBiasH2H, DEV_ERROR_MAT, DEV_ERROR,/*DEVICE*/
        nupl, TOTAL_LAYER, streams, first, GLOBAL_BIAS_SIZE , GLOBAL_W_SIZE, NSTREAMS, STREAMSIZE, NumPattern);/*VARIOUS*/

      stopAndPrint(&start, &stop);
      //cudaDeviceSynchronize();//
      
      HANDLE_CUDA(cudaMemcpy(ERROR, DEV_ERROR, sizeof(DATA), cudaMemcpyDeviceToHost));
      printf("Reduced Error: %f\n", *ERROR);	
      
      HANDLE_CUDA(cudaMemcpy(ERROR_MAT, DEV_ERROR_MAT, NumPattern*NumOutput * sizeof(DATA), cudaMemcpyDeviceToHost));
        //printMat(ERROR_MAT, NumPattern, NumOutput);
      DATA red_host = errorReductionHost(ERROR_MAT, NumPattern, NumOutput);
      printf("host reduction error : %f\n", red_host);
      /*-------------------------------------END---FEEDFORWARD-------------------------------------------*/
      /*++++-----------------------------------BACKPROPAGATION-------------------------------------------++++*/

      startTimer(&start, &stop);
      backpropagation(H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_Delta, H_H2H, H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index,
        WeightH2H, BiasH2H, DeltaWeightH2H, DeltaBiasH2H, Delta, H2H, TempDeltaWeightH2H, TempDeltaBiasH2H, 
        nupl, TOTAL_LAYER, streams, GLOBAL_BIAS_SIZE , GLOBAL_W_SIZE, GLOBAL_DELTA_SIZE, NSTREAMS, STREAMSIZE, NumPattern, eta, alpha);
      /*-------------------------------------END---BACKPROPAGATION-------------------------------------------*/
      stopAndPrint(&start, &stop);
    

      fprintf(stdout, "Epoch %-5d :   Error = %f\n", epoch, Error) ;
      if(fpl) { fprintf(fpl,"Epoch %-5d :   Error = %f\n", epoch, Error);
                  fflush(fpl); }

      if( Error < Eps ) break ;  /* stop learning when 'near enough' */
    }

#if 0
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
    for( i = 0 ; i < NumInput ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 0 ; k < NumOutput ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 0 ; p < NumPattern ; p++ ) {
    fprintf(stdout, "\n%d\t", p) ;
        for( i = 0 ; i < NumInput ; i++ ) {
            fprintf(stdout, "%f\t", Input[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(stdout, "%f\t%f\t", Target[p][k-1], Output[p][k]) ;
        }
    }
#endif
    if(verbose) {
    	printf("\nFinal Bias and Weights\n");
    }
    fp=Fopen(ResultFileName,"w");
    fpd=Fopen(DeltaFileName,"w");
    for( k = 0 ; k < NumOutput ; k ++ ) {
      if(verbose) {
        printf("Bias H to O[%d]: %f\n",k,H_BiasH2H[H_matrix_B_index[TOTAL_LAYER-2]+k]);//WeightHO[0][k]);
      }
      fprintf(fp,"%7.5f ",H_BiasH2H[H_matrix_B_index[TOTAL_LAYER-2]+k]);//WeightHO[0][k]);
      fprintf(fpd,"%g ",H_DeltaBiasH2H[H_matrix_B_index[TOTAL_LAYER-2]+k]);//DeltaWeightHO[0][k]);
      for( j = 0 ; j < nupl[NumHL] ; j++ ) {
        if(verbose) {
          printf("Weight H[%d] to O[%d]: %f\n",j,k, H_WeightH2H[H_matrix_W_index[TOTAL_LAYER-2]+ j*nupl[NumHL]+k] );//WeightHO[j][k]);
        }
        fprintf(fp,"%7.5f ", H_WeightH2H[H_matrix_W_index[TOTAL_LAYER-2]+ j*nupl[NumHL]+k]);//WeightHO[j][k]);
        fprintf(fpd,"%g ", H_DeltaWeightH2H[H_matrix_W_index[TOTAL_LAYER-2] +j*nupl[NumHL]+k]);//DeltaWeightHO[j][k]);
      }
      fprintf(fp,"\n");
      fprintf(fpd,"\n");
    }
    for(h=NumHL; h>0; h--) {
      for( j = 0 ; j < nupl[h] ; j++ ) {   
        if(verbose) {
          printf("BiasH2H[%d][%d]: %f\n",h,j, H_BiasH2H[H_matrix_B_index[h-1] +j ]);// WeightH2H[h-1][0][j]);
        }
        fprintf(fp,"%7.5f ",H_BiasH2H[H_matrix_B_index[h-1] + j]);//WeightH2H[h-1][0][j]);
        fprintf(fpd,"%g ", H_DeltaBiasH2H[H_matrix_B_index[h-1] + j]);//DeltaWeightH2H[h-1][0][j]);
        for( i = 0 ; i < nupl[h-1] ; i++ ) {
          if(verbose) {
            printf("WeightH2H[%d][%d] to H{%d]: %f\n",h,i,j, H_WeightH2H[H_matrix_W_index[h-1]+i*nupl[h]+j]);//WeightH2H[h-1][i][j]  );
          }
          fprintf(fp,"%7.5f ", H_WeightH2H[H_matrix_W_index[h-1]+i*nupl[h]+j]);//WeightH2H[h-1][i][j]);
          fprintf(fpd,"%g ", H_DeltaWeightH2H[H_matrix_W_index[h-1]+i*nupl[h]+j]);//DeltaWeightH2H[h-1][i][j]);
        }
        fprintf(fp,"\n");
        fprintf(fpd,"\n");
      }
    }
    if(fp) fclose(fp);
    if(fp) fclose(fpd);
    if(fpl) fclose(fpl);
    return 0 ;
}

/*******************************************************************************/
