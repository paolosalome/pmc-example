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

#define BOOL int
#define REAL float
#define NULLFILE "/dev/null"
#define DEFMAXEPOCH 1000
#if !defined(MAX)
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif


#define OPTIMUM_BLOCK_NUM 14 
#define BLOCK_SIDE	16 

#define OPTIMUM_BLOCK_NUM_FIRST_LAYER 4
#define BLOCK_SIDE_FIRST_LAYER 32

typedef struct grid_settings {
	unsigned int grid[3];
	unsigned int block[3];
}grid_settings;

grid_settings gs = { { OPTIMUM_BLOCK_NUM_FIRST_LAYER, OPTIMUM_BLOCK_NUM, OPTIMUM_BLOCK_NUM },{ BLOCK_SIDE_FIRST_LAYER,BLOCK_SIDE,BLOCK_SIDE } };

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
  double *dp;
  float  *sp;
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
        fscanf(fp,"%f",&(sp[i*ncol+ j]));
        break;
      case sizeof(double):
        fscanf(fp,"%lf",&(dp[i*ncol+ j]));
        break;
      default:
        writelog(TRUE,APPLICATION_RC,"invalid size in ReadFromFile: %d\n",ts);
        break;
      }
    }
  }
  fclose(fp);
}

void printMat(REAL *mat, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		printf("ROW %d : {", i);
		for (int j = 0; j < cols; j++) {
			printf("%f - ", mat[i*cols + j]);
		}
		printf("}");
		printf("\n\n");
	}
	printf("\n\n");
}

void HOST_init_struct( REAL* WeightH2H, REAL* BiasH2H, REAL* DeltaWeightH2H, REAL* DeltaBiasH2H, REAL* Delta,	int* matrix_W_index, int* matrix_B_index,  int* matrix_DELTA_index, int* matrix_H2H_index, int* nupl, int layers, int NumPattern, int smallwt) {
		
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

/*UTILITIES*/

static void HandleCuda(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_CUDA( err ) (HandleCuda( err, __FILE__, __LINE__ ))

void startTimer(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_CUDA(cudaEventCreate(start));
	HANDLE_CUDA(cudaEventCreate(stop));
	HANDLE_CUDA(cudaEventRecord(*start, 0));
}

void stopAndPrint(cudaEvent_t *start, cudaEvent_t *stop) {
	HANDLE_CUDA(cudaEventRecord(*stop, 0));
	HANDLE_CUDA(cudaEventSynchronize(*stop));
	float time = 0.0f;
	HANDLE_CUDA(cudaEventElapsedTime(&time, *start, *stop));
	printf("Elapsed Time: %f milliseconds\n", time);
	HANDLE_CUDA(cudaEventDestroy(*start));
	HANDLE_CUDA(cudaEventDestroy(*stop));
}

void optimum_grid_x(dim3* grid,int max_block,int y_limit, int width_delta){
    int x = min((width_delta+BLOCK_SIDE-1)/BLOCK_SIDE,max_block);
    int y=max_block/x;
    int prod=x*y;
    int new_prod;
    int new_x;
    for(new_x=x, new_prod=new_x*y ; new_prod != max_block && new_x > 1 && y < y_limit ;new_x--){
        y=max_block/new_x;
        new_prod=new_x*y;
    }
    if(new_prod>prod)
        x= new_x+1;
    grid->x = x;
    grid->y = y;
}       
/*DEVICE*/
texture<REAL, 2, cudaReadModeElementType> texreference_target;
__constant__ REAL alphaC[1];
__constant__ REAL etaC[1];
/*deviceReduceBlockAtomicKernel*/
__inline__ __device__ REAL warpReduceSum(REAL);
__inline__ __device__ REAL blockReduceSum(REAL);
__global__ void deviceReduceBlockAtomicKernel(REAL *, REAL*, int);

/*MMMul(for feedforward)*/
__device__ void MMMulDevPartialFeed(REAL *, REAL *, REAL *, REAL *, REAL*, REAL *, unsigned int, unsigned int, unsigned int, unsigned int, int);
__global__ void MMMulDevFeed(REAL *, REAL *, REAL *, REAL *, REAL *, REAL*, unsigned int, unsigned int, unsigned int, int, int);

/* MMMul backpropagation*/
__device__ void MMMulDevPartialBack(REAL* , REAL* , REAL* , REAL* , REAL* , REAL* , int , int , int , int , BOOL, int, int);
__global__ void MMMulDevBack(REAL* , REAL* , REAL* , REAL* ,REAL* , REAL* , int , int , int , int , BOOL, int, int);
/* MMMul reduction after backpropagation*/
__device__ void MMMulReductionBlock(REAL* , REAL* , REAL* , REAL* , REAL* , REAL* , int , int , int , int ,  int , int , BOOL, int);
__global__ void MMMulReduction(REAL* , REAL* , REAL* , REAL* , REAL* , REAL* , int , int , int , int ,  int , int , BOOL, int);


/*HOST*/
void feedforward(REAL *, REAL* , REAL* ,REAL*,REAL*, int* , int*, int* , int* ,
	REAL* , REAL* , REAL* , REAL* , REAL*, REAL*,
	REAL *, REAL *, int *, int , cudaStream_t *, BOOL,
	int  , int , int , int , int );
void backpropagation(REAL* , REAL* , REAL* , REAL* , REAL* , REAL* , int* , int*,  int* , int* ,
	REAL* , REAL* , REAL* , REAL* , REAL* ,REAL* ,REAL* , REAL* , 
	int*  , int , cudaStream_t* ,int  , int , int , int , int , int);


int main(int argc, char *argv[]) {
    int     h, i, j, k, epoch;
    int    NumPattern, NumInput, NumOutput;
    /* device and host matrix */
    REAL* WeightH2H, *BiasH2H, *DeltaWeightH2H, *DeltaBiasH2H, *Delta, *H2H, *TempDeltaWeightH2H, *TempDeltaBiasH2H;
    REAL* H_WeightH2H,*H_BiasH2H,*H_DeltaWeightH2H,*H_DeltaBiasH2H,*H_Delta,*H_H2H;
    

    REAL *INPUT_MAT, *ERROR_MAT, *DEV_ERROR_MAT;
    REAL *ERROR, *DEV_ERROR;
    REAL *TARGET;
    int NSTREAMS;
    REAL Eps, eta, alpha, smallwt;
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
    NSTREAMS=3;
    nupl=(int *)makevect(NumHL+2,sizeof(int));
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
  
    cudaStream_t streams[NSTREAMS];
    int TOTAL_LAYER = NumHL +2; //#of layers
    int STREAMSIZE = NumPattern/NSTREAMS;
    /* total size of h2h,delta,w,bias arrays */
		int GLOBAL_H_SIZE= NumPattern*NumInput, GLOBAL_DELTA_SIZE=0, GLOBAL_W_SIZE=0, GLOBAL_BIAS_SIZE=0;
    for(int numl=1;numl<TOTAL_LAYER;numl++){
      GLOBAL_H_SIZE += nupl[numl]*NumPattern;
      GLOBAL_DELTA_SIZE += nupl[numl]*NumPattern;
      GLOBAL_W_SIZE += nupl[numl-1]*nupl[numl];
      GLOBAL_BIAS_SIZE += nupl[numl];
    }
/*++++------------------------------------ERRORS--------------------------------------------------++++*/

    ERROR_MAT = (REAL*)malloc(NumPattern*NumOutput * sizeof(REAL)); // ERROR FOR CHECKING CORRECTNESS
    HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR_MAT, NumPattern*NumOutput * sizeof(REAL))); //DEVICE ERROR MAT

    ERROR = (REAL*)malloc(sizeof(REAL)); // ERROR FOR CHECKING CORRECTNESS
    HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR, sizeof(REAL))); //DEVICE ERROR
    HANDLE_CUDA(cudaMemset(DEV_ERROR, 0, sizeof(REAL)));

    /*----------------------------------------ERRORS END--------------------------------------------------*/
  /*++++---------------------------init INPUT_MAT and TARGET (HOST)-----------------------------++++*/
    
    TARGET = (REAL*)malloc(NumOutput*NumPattern * sizeof(REAL)); //TARGET OF THE PATTERNS
    /*INPUT_MAT is pinned memory*/  
    HANDLE_CUDA(cudaHostAlloc(&INPUT_MAT, NumInput * NumPattern * sizeof(REAL), 0));

    ReadFromFileArr(InputFileName,INPUT_MAT,NumPattern,NumInput,sizeof(REAL));
    ReadFromFileArr(TargetFileName,TARGET,NumPattern,NumOutput,sizeof(REAL));

	/*---------------------------end init INPUT_MAT and TARGET (HOST)-------------------------*/

  /*++++---------------------------REAL structures on host and device-------------------------++++*/
	/*	HOST MATRIX AND INDEXES FOR PADDING */
    H_WeightH2H = (REAL*)makevect(GLOBAL_W_SIZE,sizeof(REAL));
    H_BiasH2H = (REAL*)makevect(GLOBAL_BIAS_SIZE,sizeof(REAL));
    H_DeltaWeightH2H = (REAL*)makevect(GLOBAL_W_SIZE,sizeof(REAL));
    H_DeltaBiasH2H = (REAL*)makevect(GLOBAL_BIAS_SIZE,sizeof(REAL));
    H_Delta = (REAL*)makevect(GLOBAL_DELTA_SIZE,sizeof(REAL));
    H_H2H = (REAL*)makevect(GLOBAL_H_SIZE,sizeof(REAL));

    int H_matrix_W_index[TOTAL_LAYER - 1];//INDEX for padding in Weight 
    int H_matrix_B_index[TOTAL_LAYER - 1];//INDEX for padding in Bias 
    int H_matrix_DELTA_index[TOTAL_LAYER - 1];//INDEX for padding in Delta
    int H_matrix_H2H_index[TOTAL_LAYER];//INDEX for padding in H2H

    //Init weights and biases on host
    HOST_init_struct(H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_Delta, H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index, nupl, TOTAL_LAYER, NumPattern, smallwt);
    //    DEVICE    Malloc the necessary space on device memory
    HANDLE_CUDA(cudaMalloc((void **)&WeightH2H , GLOBAL_W_SIZE * sizeof(REAL)));
    HANDLE_CUDA(cudaMalloc((void **)&BiasH2H, GLOBAL_BIAS_SIZE * sizeof(REAL)));
    HANDLE_CUDA(cudaMalloc((void **)&DeltaWeightH2H, GLOBAL_W_SIZE * sizeof(REAL)));
    HANDLE_CUDA(cudaMalloc((void **)&DeltaBiasH2H, GLOBAL_BIAS_SIZE * sizeof(REAL)));
    HANDLE_CUDA(cudaMalloc((void **)&Delta, GLOBAL_DELTA_SIZE * sizeof(REAL)));
    HANDLE_CUDA(cudaMalloc((void **)&H2H, GLOBAL_H_SIZE * sizeof(REAL)));
    HANDLE_CUDA(cudaMalloc((void **)&TempDeltaWeightH2H, NSTREAMS*GLOBAL_W_SIZE *sizeof(REAL)));
    HANDLE_CUDA(cudaMalloc((void **)&TempDeltaBiasH2H, NSTREAMS*GLOBAL_BIAS_SIZE *sizeof(REAL)));

    HANDLE_CUDA(cudaMemcpyToSymbol(etaC,&eta,sizeof(REAL)));
    HANDLE_CUDA(cudaMemcpyToSymbol(alphaC,&alpha,sizeof(REAL)));
  /*---------------------------end REAL structures on host and device----------------------------*/
  /*++++---------------------------cuda array for texture-----------------------------++++*/
    cudaArray* DEV_TARGET_CUDA;
    cudaChannelFormatDesc channel;

    channel = cudaCreateChannelDesc<REAL>();
    HANDLE_CUDA(cudaMallocArray(&DEV_TARGET_CUDA, &channel, NumOutput, NumPattern));
    HANDLE_CUDA(cudaMemcpyToArray(DEV_TARGET_CUDA, 0, 0, TARGET, NumOutput*NumPattern * sizeof(REAL), cudaMemcpyHostToDevice));

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
    /*++++---------------------------------Scan files----------------------------------++++*/

    if(strcmp(LogFileName,NULLFILE)) { fpl=Fopen(LogFileName,"w"); }

    if(strcmp(RestartFileName,NULLFILE)) {
        fp=Fopen(RestartFileName,"r");
        if(strcmp(RestartDeltaFileName,NULLFILE)) {
                fpd=Fopen(RestartDeltaFileName,"r");
        }
        for( k = 0 ; k < nupl[NumHL+1] ; k ++ ) {
          
                fscanf(fp,formatstring,&H_BiasH2H[H_matrix_B_index[TOTAL_LAYER-2]+k]);//WeightHO[0][k]);
                if(fpd)
                  fscanf(fpd,formatstring,&H_DeltaBiasH2H[H_matrix_B_index[TOTAL_LAYER-2]+k]);//WeightHO[0][k]);
                
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
    fprintf(stderr,"end reading\n");
    /*---------------------------------end Scan files----------------------------------*/
    /*++++---------------------------------Iteration throught epoch----------------------------------++++*/

    cudaEvent_t start, stop;
    for( epoch = 0 ; epoch < maxepoch ; epoch++) {    /* iterate weight updates */

     /*++++-----------------------------------FEEDFORWARD-------------------------------------------++++*/

    
      printf("\nEPOCH %d\n",epoch+1);
      startTimer(&start, &stop);
      BOOL first=(epoch==0)?1:0;
      feedforward(INPUT_MAT,H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index, /*HOST*/
        WeightH2H, BiasH2H, Delta, H2H, DeltaWeightH2H, DeltaBiasH2H, DEV_ERROR_MAT, DEV_ERROR,/*DEVICE*/
        nupl, TOTAL_LAYER, streams, first, GLOBAL_BIAS_SIZE , GLOBAL_W_SIZE, NSTREAMS, STREAMSIZE, NumPattern);/*VARIOUS*/

      stopAndPrint(&start, &stop);
      
      HANDLE_CUDA(cudaMemcpy(ERROR, DEV_ERROR, sizeof(REAL), cudaMemcpyDeviceToHost));
      /*-------------------------------------END---FEEDFORWARD-------------------------------------------*/
      /*++++-----------------------------------BACKPROPAGATION-------------------------------------------++++*/

      startTimer(&start, &stop);
      backpropagation(H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_Delta, H_H2H, H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index,
        WeightH2H, BiasH2H, DeltaWeightH2H, DeltaBiasH2H, Delta, H2H, TempDeltaWeightH2H, TempDeltaBiasH2H, 
        nupl, TOTAL_LAYER, streams, GLOBAL_BIAS_SIZE , GLOBAL_W_SIZE, GLOBAL_DELTA_SIZE, NSTREAMS, STREAMSIZE, NumPattern);
      /*-------------------------------------END---BACKPROPAGATION-------------------------------------------*/
      stopAndPrint(&start, &stop);
    

      fprintf(stdout, "Epoch %-5d :   Error = %f\n", epoch, *ERROR) ;
      if(fpl) { fprintf(fpl,"Epoch %-5d :   Error = %f\n", epoch, *ERROR);
                  fflush(fpl); }

      if( *ERROR < Eps ) break ;  /* stop learning when 'near enough' */
      HANDLE_CUDA(cudaMemset(DEV_ERROR_MAT, 0, sizeof(REAL)));

      HANDLE_CUDA(cudaMemset(DEV_ERROR, 0, sizeof(REAL)));

    }
  /*---------------------------------end Iteration throught epoch----------------------------------*/

    HANDLE_CUDA(cudaMemcpy( H_DeltaWeightH2H , DeltaWeightH2H, GLOBAL_W_SIZE*sizeof(REAL), cudaMemcpyDeviceToHost));
    HANDLE_CUDA(cudaMemcpy( H_DeltaBiasH2H, DeltaBiasH2H, (GLOBAL_BIAS_SIZE)*sizeof(REAL), cudaMemcpyDeviceToHost));
    HANDLE_CUDA(cudaMemcpy( H_WeightH2H , WeightH2H, GLOBAL_W_SIZE*sizeof(REAL), cudaMemcpyDeviceToHost));
    HANDLE_CUDA(cudaMemcpy( H_BiasH2H, BiasH2H, (GLOBAL_BIAS_SIZE)*sizeof(REAL), cudaMemcpyDeviceToHost));
	

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


  /*++++--------------------------------deallocations------------------------------------++++*/
  
    //Host dealloc
    free(nupl);
    free(TARGET);
    free(ERROR_MAT);
    free(ERROR);
    free(H_WeightH2H);
    free(H_BiasH2H);
    free(H_DeltaWeightH2H);
    free(H_DeltaBiasH2H);
    free(H_Delta);
    free(H_H2H);
    //Device dealloc
    cudaFreeHost(INPUT_MAT);
    cudaFree(DEV_ERROR_MAT);
    cudaFree(DEV_ERROR);
    cudaFree(WeightH2H);
    cudaFree(BiasH2H);
    cudaFree(DeltaWeightH2H);
    cudaFree(DeltaBiasH2H);
    cudaFree(Delta);
    cudaFree(H2H);
    cudaFree(TempDeltaWeightH2H);
    cudaFree(TempDeltaBiasH2H);
    
    //Unbinding texture
    cudaUnbindTexture(texreference_target);
    //Free cuda array
    cudaFreeArray(DEV_TARGET_CUDA);

    /*------------------------------------end deallocations------------------------------------*/

    return 0 ;
}

/*******************************************************************************/



/*---------------------------------------------------------------------KERNEL--------------------------------------------------------------------------*/


/*++++---------------------------deviceReduceBlockAtomicKernel---------------------------++++*/

/*Warp reduction*/
__inline__ __device__ REAL warpReduceSum(REAL val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

/*Block reduction*/
__inline__ __device__ REAL blockReduceSum(REAL val) {

	static __shared__ REAL shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);

	if (lane == 0) shared[wid] = val;

	__syncthreads();

	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) val = warpReduceSum(val);

	return val;
}

/*Reducing large arrays--Blocks implementation*/

//Nella chiamata di questo kernel è meglio usare una griglia lineare di 8 blocchi con 256 threads ciascuno -- 
//In tal modo vengono limitati gli accessi alla shared memory rispetto all'implementazione con 2 blocchi da 1024 threads ciascuno
//Attenzione ai possibili arrotondamenti di numeri a virgola mobile dovuti alle atomicAdd.
__global__ void deviceReduceBlockAtomicKernel(REAL *in, REAL* out, int N) {
	REAL sum = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x ; i < N ; i += blockDim.x * gridDim.x) {
		sum += in[i];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0)
		atomicAdd(out, sum);
}

/*-------------------------------end--deviceReduceBlockAtomicKernel--------------------------*/

/*++++---------------------------MMMul--Feedforward-------------------------++++*/

/* h2h è il puntatore alla porzione dell'h2h globale da considerare in questa fase
(ad ogni passo il kernel che invoca questo device incrementa il puntatore h2h
in modo proporzionale al patt_per_step (e similmente h2h_dest) (vedi sotto)).
offset_y è la posizione considerata lungo le y (nelle matrici h2h, h2h_dest ed eventualmente error) durante la chiamata corrente a __device__.
Delta è calcolato per l'output layer (propagato poi con backpropagation) --> DeltaO[p][k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;
*/

__device__ void MMMulDevPartialFeed(REAL *h2h, REAL *w, REAL *biases, REAL *h2h_dest, REAL *delta, REAL *error, unsigned int row_w, unsigned int col_w, unsigned int num_pattern, unsigned int offset_y, int NumOutput) {

	int tx = threadIdx.x, ty = threadIdx.y;
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	const int block_dim = blockDim.x; // assumiamo che i blocchi siano quadrati
	int dest_x = block_x*block_dim + tx;
	int dest_y = block_y*block_dim + ty;

	int w_x = block_x*block_dim; // start block in w
	int h2h_y = block_y*block_dim*row_w; // start block in h2h

	int end_h2h = h2h_y + row_w - 1; // last block position in h2h

	int step_w = block_dim*col_w;
	int step_h2h = block_dim;
	int min;

	REAL partial = 0.0f;
	int block_r_border = 0; // contatore che indica in che iterazione dei blocchi ci troviamo
	int current_inc;

	for (int wid = w_x, h2h_id = h2h_y; h2h_id <= end_h2h; wid += step_w, h2h_id += step_h2h) {

		block_r_border += block_dim;

		__shared__ REAL shared_w[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];
		__shared__ REAL shared_h2h[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];

		int t_index_w = wid + tx + ty*col_w;
		int t_index_h2h = h2h_id + tx + ty*row_w;

		shared_h2h[ty][tx] = (t_index_h2h < num_pattern*row_w) ? (h2h[t_index_h2h]) : (0.0f);
		shared_w[ty][tx] = (t_index_w < col_w*row_w) ? (w[t_index_w]) : (0.0f);

		__syncthreads();

		current_inc = row_w - (block_r_border - block_dim);

		min = (current_inc < block_dim) ? (current_inc) : (block_dim);

		//#pragma unroll(2)
		for (int k = 0; k < min; k++) {
			partial += shared_h2h[ty][k] * shared_w[k][tx];
		}

		__syncthreads();
	}

	if (dest_x < col_w && dest_y < num_pattern) {

		REAL out = 1.0 / (1.0 + exp(-(partial + biases[dest_x])));
		h2h_dest[dest_y*col_w + dest_x] = out; //SIGMA

		//Se siamo nell'ultimo passo
		if (col_w == NumOutput) {
			
			REAL target = tex2D(texreference_target, dest_x, dest_y + offset_y);

			//Scrivi nella posizione corrispondente della matrice di ERRORE
			/*0.5*(Target[p][k] - Output[p][k])*(Target[p][k] - Output[p][k])*/
			error[dest_y*col_w + dest_x] =  (REAL)0.5*(target - out)*(target - out);

			//Scrivi nella posizione corrispondente della matrice DELTA
			/*(Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k])*/
			delta[dest_y*col_w + dest_x] = (REAL)(target - out)*(out)*(1.0- out);
		}
	}
}

/*patt_per_step è il numero di pattern (quando possibile...) da considerare in ciascuna iterazione su h2h*/
/*Questo kernel ad ogni passo incrementa il puntatore ad h2h di num_patt_per_step*NEURO_L_L_1 (e similmente h2h_dest),
controlla che sia ancora nel range di h2h, e calcola num_pattern (vedi sopra) in funzione dei
pattern mancanti.
stream_offset_y è la posizione lungo le y da cui parte (nelle matrici h2h e h2h_dest) lo stream corrente.
*/
//Dove ora c'� STREAMSIZE prima c'era NumPattern
__global__ void MMMulDevFeed(REAL *h2h, REAL *w, REAL *biases, REAL *h2h_dest, REAL *delta, REAL *error, unsigned int row_w, unsigned int col_w, unsigned int stream_offset_y,
						int STREAMSIZE, int NumOutput) {

  unsigned int patt_per_step = gridDim.y*blockDim.y;
	unsigned int current_patts;
	unsigned int remaining_patts;
	const int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
												   //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)		

	for (unsigned int y = 0; y < STREAMSIZE; y += patt_per_step) {

		remaining_patts = STREAMSIZE - y;
		current_patts = (remaining_patts < patt_per_step) ? (remaining_patts) : (patt_per_step);

		if (pos_block_y >= current_patts) { return; }

		MMMulDevPartialFeed(h2h + y*row_w, w, biases, h2h_dest + y*col_w, delta + y*NumOutput, error + y*NumOutput, row_w, col_w, current_patts, stream_offset_y + y, NumOutput);
	}
}

/*-------------------------------end--MMMul--Feedforward------------------------*/

/*-------------------------------MMul--Backpropagation--------------------------*/
/* 
Si calcola una porzione della matrice delta_W,delta_bias e di delta_h2h(dest_delta) a livello di blocco. 
La griglia è adagiata sulla matrice delta_W (di conseguenza anche su W che ha le stesse dimensioni e offset di delta_W) ed ogni
blocco si occupa della sua porzione senza interferenza con altri blocchi della stessa. La posizione del blocco all'interno della griglia determina quale 
colonne di larghezza BLOCK_SIDE considerare nelle matrici delta_h2h(del livello successivo), h2h(di quello corrente): la prima dipende da b_x 
mentre la seconda da b_y(cosi come dest_delta che ha le stesse dimensioni di h2h). Tale offset è determinato all'invocazione di questo kernel.
*/
__device__ void MMMulDevPartialBack(REAL* h2h, REAL* w, REAL* delta, REAL* dest_delta, REAL* delta_weight_dest, REAL* delta_bias_dest, int width_h2h, int width_delta, int h2h_right_limit, int delta_right_limit, BOOL enable_bias, int layer, int STREAMSIZE){
  int t_x = threadIdx.x;
  int t_y = threadIdx.y;
    /* int idx = t_x + blockIdx.x*BLOCK_SIDE ; 
    int idy = t_y + blockIdx.y*BLOCK_SIDE ;  */
  int pattern;
  int h2h_corner = blockIdx.y*BLOCK_SIDE;
  int delta_corner = blockIdx.x*BLOCK_SIDE;

  //__shared__ REAL temp_sum_delta_h2h[BLOCK_SIDE*BLOCK_SIDE];
  __shared__ REAL block_h2h[BLOCK_SIDE*BLOCK_SIDE];
  __shared__ REAL block_w[BLOCK_SIDE*(BLOCK_SIDE)];
  __shared__ REAL block_delta[BLOCK_SIDE*BLOCK_SIDE];
  __shared__ REAL bias_to_update[BLOCK_SIDE*BLOCK_SIDE];

  int max_b_x = ((delta_corner + BLOCK_SIDE) < delta_right_limit) ? BLOCK_SIDE: (delta_right_limit - delta_corner);
  int max_a_x = ((h2h_corner + BLOCK_SIDE) < h2h_right_limit) ? BLOCK_SIDE: (h2h_right_limit - h2h_corner);
  REAL thr_deltaW[BLOCK_SIDE];
  for(int j =0 ;j<BLOCK_SIDE;j++)
    thr_deltaW[j]=0.0;
  //temp_sum_delta_h2h[t_x+t_y*BLOCK_SIDE]=0.0f;
  //trasposta di W caricata una sola volta 
	if(layer>0)
	  block_w[t_x*BLOCK_SIDE+t_y] = (max_a_x > t_y && max_b_x > t_x) ? w[t_y*width_delta + t_x]:0.0f;

  if(enable_bias==1)
    bias_to_update[t_y*BLOCK_SIDE + t_x] = 0.0f;

  for(int curr_patterns=0;curr_patterns<STREAMSIZE;curr_patterns+=BLOCK_SIDE){
    pattern = (curr_patterns  + BLOCK_SIDE > STREAMSIZE) ? (STREAMSIZE-curr_patterns): BLOCK_SIDE ;

    block_h2h[t_y*BLOCK_SIDE+t_x]= ((curr_patterns+ t_y) < STREAMSIZE && max_a_x > t_x) ? h2h[t_y*width_h2h + t_x +curr_patterns*width_h2h]:0.0f;
		block_delta[t_y*BLOCK_SIDE+t_x] = ((curr_patterns+ t_y) < STREAMSIZE && max_b_x > t_x) ? delta[t_y*width_delta + t_x + curr_patterns*width_delta]:0.0f;
    REAL val = block_delta[t_y*BLOCK_SIDE+t_x];
    REAL temp=0.0f;
        
    __syncthreads();
		for(int i=0 ;i<BLOCK_SIDE;i++){
			if(layer>0)
        temp += block_delta[t_y*BLOCK_SIDE+i]*block_w[i*BLOCK_SIDE+t_x];//product ROW-COLUMN delta*W by trd[ty][tx]
      thr_deltaW[i]+=val*block_h2h[t_y*BLOCK_SIDE+i];
      /*ogni thread si costruisce una porzione di deltaW salvandola in un array locale(la riduzione di tale  )*/
      
      /*ogni thread del blocco contribuisce a deltaW (i thread con stesso t_x contribuiscono allo stesso elemento)
          atomicAdd(&temp_sum_delta_h2h[t_x+i*BLOCK_SIDE],val*block_h2h[t_y*BLOCK_SIDE+i]); 
          */
    }    
      
    /*L'interferenza rimane nel calcolo del delta_h2h del livello corrente (dest_delta) tra blocchi adiacenti lungo la larghezza della griglia.*/
    if(layer > 0 && t_y < pattern)
      atomicAdd(&dest_delta[t_y*width_h2h+ curr_patterns*width_h2h + t_x], temp*block_h2h[t_y*BLOCK_SIDE+t_x]*(1.0-block_h2h[t_y*BLOCK_SIDE+t_x]));//product 
   
    if(enable_bias==1)//solo i blocchi con blocky = 0
      bias_to_update[t_y*BLOCK_SIDE + t_x] += block_delta[t_y*BLOCK_SIDE + t_x];
  }
  /*si aggiorna deltaW dello stream*/
  /*for(int i=0;i<BLOCK_SIDE;i++)
    atomicAdd(&temp_sum_delta_h2h[t_x+i*BLOCK_SIDE],thr_deltaW[i]);
  __syncthreads();*/
  for(int i=0;i<BLOCK_SIDE;i++){
    if( (i + h2h_corner) < h2h_right_limit && (t_x + delta_corner) < delta_right_limit){
    //if( (t_y + h2h_corner) < h2h_right_limit && (t_x + delta_corner) < delta_right_limit){
      //delta_weight_dest[t_x+t_y*width_delta] = etaC[0]*temp_sum_delta_h2h[t_y*BLOCK_SIDE+ t_x];
      atomicAdd(&delta_weight_dest[t_x+i*width_delta],etaC[0]*thr_deltaW[i]);
    }
  }
  /* si aggiorna deltaBias dello stream */
  if(enable_bias==1 &&  t_y==0 && (t_x + delta_corner) < delta_right_limit){
    REAL tempBias=0.0f;
    for(int i=0;i<BLOCK_SIDE;i++)
      tempBias+=bias_to_update[i*BLOCK_SIDE+t_x];
    delta_bias_dest[t_x] = etaC[0]*tempBias ; 
  } 
}


/* si effettua la riduzione finale di DELTA_W sommando i delta_w calcolati in ogni stream.
  ogni thread del blocco lavora in modo indipendente dagli altri caricando solo le celle  che gli competono per poi aggiornare 
  DELTA_WEIGHT,DELTA_BIAS,W e BIAS  */
__device__ void MMMulReductionBlock(REAL* W, REAL* BIAS, REAL* DELTA_WEIGHT, REAL* DELTA_BIAS, REAL* DELTA_WEIGHT_DEST, REAL* DELTA_BIAS_DEST, int offset_weight, int offset_bias, int width_h2h, int width_delta,  int Y_right_limit, int X_right_limit, BOOL enable_bias, int NSTREAMS){
    
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int b_x = blockIdx.x*BLOCK_SIDE;
    int b_y = blockIdx.y*BLOCK_SIDE;
    int offset_block_w = b_x+b_y*width_delta;
    int offset_block_bias = b_x;
 
    if( (b_x + t_x) < X_right_limit && (b_y+ t_y) < Y_right_limit){
       
        REAL dw_loc = alphaC[0]*DELTA_WEIGHT[t_x+t_y*width_delta];
        REAL dbias_loc;
                
        for(int i=0 ;i<NSTREAMS; i++){
            dw_loc += DELTA_WEIGHT_DEST[i*width_delta*width_h2h+ offset_weight + offset_block_w + t_x + t_y*width_delta];//offset della griglia + offset blocco + offset thr
            if(enable_bias==1 && t_y==0){
                dbias_loc += DELTA_BIAS_DEST[i*width_delta+ offset_bias + offset_block_bias + t_x];
            }
        }
        if(enable_bias==1 && t_y==0){
            BIAS[t_x] += dbias_loc + alphaC[0]*DELTA_BIAS[t_x];
            DELTA_BIAS[t_x] = dbias_loc + alphaC[0]*DELTA_BIAS[t_x];
        }
        W[t_x + t_y*width_delta] += dw_loc;
        DELTA_WEIGHT[t_x + t_y*width_delta] = dw_loc;
    }
}
/* se il blocco è entro i limiti della matrice deltaW può effetuare la riduzione della parte di deltaW che gli compete 
 Le matrici DELTA_WEIGHT e DELTA_BIAS sono le matrici presenti nell'epoca precedente che devono essere aggiornate, così come W e BIAS,
 riducendo prima le matrici "temporanee" DELTA_WEIGHT_DEST e DELTA_BIAS_DEST
*/
__global__ void MMMulReduction(REAL* W, REAL* BIAS, REAL* DELTA_WEIGHT, REAL* DELTA_BIAS, REAL* DELTA_WEIGHT_DEST, REAL* DELTA_BIAS_DEST, int offset_weight, int offset_bias, int width_h2h, int width_delta,  int Y_right_limit, int X_right_limit, BOOL enable_bias, int NSTREAMS){
    int b_x = blockIdx.x*BLOCK_SIDE;
    int b_y = blockIdx.y*BLOCK_SIDE;
    //enable bias vale 1 se la griglia si è spostata lungo la x . Gli unici blocchi che calcoleranno il delta bias sono quelli con blockIdy = 0
    if(b_x < X_right_limit && b_y <Y_right_limit)
      MMMulReductionBlock(W+b_x+b_y*width_delta, BIAS+ b_x, DELTA_WEIGHT+b_x+b_y*width_delta, DELTA_BIAS+ b_x, DELTA_WEIGHT_DEST, DELTA_BIAS_DEST, offset_weight, offset_bias, width_h2h, width_delta,  Y_right_limit, X_right_limit, enable_bias*(1-blockIdx.y),NSTREAMS);
    //__syncthreads();
}
/* se il blocco è entro i limiti della matrice deltaW può calcolare la parte di deltaW che gli compete
Le matrici DELTA_WEIGHT_DEST e DELTA_BIAS_DEST sono delle matrici "temporanee" di dimensione pari a NSTREAMS * la dimensione di DELTA_W e DELTA_BIAS.
Ad ogni stream viene associata la porzione corrispondente di tali matrici alla chiamata del kernel di dimensione pari a DELTA_W (e DELTA_Bias)
*/
__global__ void MMMulDevBack(REAL* H2H, REAL* W, REAL* DELTA, REAL* DEST_DELTA, REAL* DELTA_WEIGHT_DEST, REAL* DELTA_BIAS_DEST, int width_h2h, int width_delta, int h2h_right_limit, int delta_right_limit, BOOL enable_bias,int layer, int STREAMSIZE){
    int b_x = blockIdx.x*BLOCK_SIDE;
    int b_y = blockIdx.y*BLOCK_SIDE;
    //enable bias vale 1 se la griglia si è spostata lungo la x . Gli unici blocchi che calcoleranno il delta bias sono quelli con blockIdy = 0
	
	  if(b_x < delta_right_limit && b_y <h2h_right_limit)
      MMMulDevPartialBack(H2H +b_y, W +b_x+b_y*width_delta, DELTA +b_x, DEST_DELTA+b_y, DELTA_WEIGHT_DEST +b_x+b_y*width_delta, DELTA_BIAS_DEST +b_x, width_h2h, width_delta, h2h_right_limit, delta_right_limit, enable_bias*(1-blockIdx.y),layer, STREAMSIZE);
    //__syncthreads();
}
/*-----------------------------end--MMul--Backpropagation--------------------------*/


/*HOST*/

/*FIRT PHASE OF THE ALGORITHM -- THE INPUT IS TRANSMITTED VIA THE NETWORK*/ 
void feedforward (REAL *INPUT, 
	REAL* H_WeightH2H, REAL* H_BiasH2H, REAL* H_DeltaWeightH2H, REAL* H_DeltaBiasH2H, int* H_matrix_W_index, int* H_matrix_B_index, int* H_matrix_DELTA_index, int* H_matrix_H2H_index,
	REAL* WeightH2H, REAL* BiasH2H, REAL* Delta, REAL* H2H, REAL* DeltaWeightH2H, REAL* DeltaBiasH2H,
	REAL *dev_error_mat, REAL *dev_error, int *nupl, int layers, cudaStream_t *streams, BOOL first_epoch,
	int GLOBAL_BIAS_SIZE , int GLOBAL_W_SIZE, int NSTREAMS, int STREAMSIZE, int NumPattern) {
	//cudaEvent_t start, stop;

	//Grid setting
	dim3 grid, block;
	//Useful pointers
	REAL *h2h, *w, *bias, *h2h_dest, *delta, *error;

	int offset,strsz= STREAMSIZE;
  /* solo nella prima epoca si effettua la copia dei dati delle matrici W,DeltaW,Bias,DeltaBias*/
	if (first_epoch == 1) {
		HANDLE_CUDA(cudaMemcpy(WeightH2H, H_WeightH2H, GLOBAL_W_SIZE * sizeof(REAL), cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(BiasH2H, H_BiasH2H, GLOBAL_BIAS_SIZE * sizeof(REAL), cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(DeltaWeightH2H, H_DeltaWeightH2H, GLOBAL_W_SIZE * sizeof(REAL), cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(DeltaBiasH2H, H_DeltaBiasH2H, GLOBAL_BIAS_SIZE * sizeof(REAL), cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < NSTREAMS; i++) {

		block.x = gs.block[0];
		block.y = gs.block[0];
		grid.x = (nupl[1] + block.x - 1) / block.x;
		grid.y = gs.grid[0] / grid.x;

    /*ogni stream ha il suo offset, ma non è detto che il numero di patterns sia divisibile per il numero di stream;
     quindi l'ultimo potrebbe averne meno */
    offset = i*STREAMSIZE;
    strsz += (i==2)?NumPattern%(NSTREAMS*STREAMSIZE):0;

    /* si settano i puntatori del layer corrente*/
    h2h = H2H + offset*nupl[0];
		w = WeightH2H;
		bias = BiasH2H;
		h2h_dest = H2H + H_matrix_H2H_index[1] + offset*nupl[1];
		delta = Delta + H_matrix_DELTA_index[layers - 2] + offset*nupl[layers - 1];
		error = dev_error_mat + offset*nupl[layers - 1];
   
		if (first_epoch == 1) {
			HANDLE_CUDA(cudaMemcpyAsync(h2h, INPUT + offset*nupl[0], nupl[0] * STREAMSIZE * sizeof(REAL), cudaMemcpyHostToDevice, streams[i]));
		}

		MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, error, nupl[0], nupl[1], offset, strsz, nupl[layers-1]);

		for (int l = 1; l < (layers - 1); l++) {

			block.x = gs.block[l];
			block.y = gs.block[l];
			grid.x = (nupl[l + 1] + block.x - 1) / block.x;
      grid.y = gs.grid[l] / grid.x;
      
			/* si settano i puntatori del layer corrente*/
			h2h = H2H + H_matrix_H2H_index[l] + offset*nupl[l];
			w = WeightH2H + H_matrix_W_index[l];
			bias = BiasH2H + H_matrix_B_index[l];
			h2h_dest = H2H + H_matrix_H2H_index[l + 1] + offset*nupl[l + 1];
			//Delta and error already set up

			MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, error, nupl[l], nupl[l + 1], offset, STREAMSIZE, nupl[layers-1]);
		}
	}
	//Error reduction (default stream)
	deviceReduceBlockAtomicKernel << <OPTIMUM_BLOCK_NUM * 2, BLOCK_SIDE*BLOCK_SIDE >> > (dev_error_mat, dev_error, NumPattern*nupl[layers-1]);
}

/*	SECONDA FASE DELL'ALGORITMO : L'ERRORE E' PROPAGATO A RITROSO LUNGO LA RETE */
void backpropagation(REAL* H_WeightH2H, REAL* H_BiasH2H, REAL* H_DeltaWeightH2H, REAL* H_DeltaBiasH2H, REAL* H_Delta, REAL* H_H2H, int* H_matrix_W_index, int * H_matrix_B_index,int* H_matrix_DELTA_index, int* H_matrix_H2H_index,
	REAL* WeightH2H, REAL* BiasH2H, REAL* DeltaWeightH2H, REAL* DeltaBiasH2H, REAL* Delta,REAL* H2H,REAL* TempDeltaWeightH2H, REAL* TempDeltaBiasH2H, 
	int* nupl , int layers, cudaStream_t* streams,int GLOBAL_BIAS_SIZE , int GLOBAL_W_SIZE, int GLOBAL_DELTA_SIZE, int NSTREAMS, int STREAMSIZE, int NumPattern){
    
  dim3 grid,block;
  block.x= BLOCK_SIDE;
  block.y= BLOCK_SIDE;

	REAL *d_h2h, *d_w, *d_bias, *d_delta_weight, *d_delta_bias, *d_delta, *d_dest_delta, *d_delta_weight_dest, *d_delta_bias_dest;
  /*ogni stream ha il suo offset, ma non è detto che il numero di patterns sia divisibile per il numero di stream;
  quindi l'ultimo potrebbe averne meno */
  int offset,strsz=STREAMSIZE;

  /*  BACKPROPAGATION (SU PIU' STREAM) */
	for(int str=0;str<NSTREAMS;str++){
		offset=str*STREAMSIZE;
    strsz += (str==2)?NumPattern%(NSTREAMS*STREAMSIZE):0;

		for (int l = (layers -2); l > 0; l--) {
      /*  si imposta la griglia in modo da "coprire" la matrice deltaW su cui si adagia*/
			optimum_grid_x(&grid, OPTIMUM_BLOCK_NUM, nupl[l]/BLOCK_SIDE, nupl[l + 1]);
			/* si settano i puntatori del layer corrente*/
			d_h2h = H2H + H_matrix_H2H_index[l] + offset*nupl[l];
			d_w = WeightH2H + H_matrix_W_index[l];
			d_delta_weight_dest = TempDeltaWeightH2H + NSTREAMS*H_matrix_W_index[l];
			d_delta_bias_dest = TempDeltaBiasH2H + NSTREAMS*H_matrix_B_index[l];
			d_delta = Delta + H_matrix_DELTA_index[l] + offset*nupl[l+1];
			d_dest_delta = Delta + H_matrix_DELTA_index[l-1] + offset*nupl[l];
      
      /* la dimensione della griglia è fissata, per tanto bisogna "spostarla" lungo la matrice deltaW */
			for(int sw_x=0; sw_x < nupl[l+1]; sw_x += grid.x*BLOCK_SIDE){
				for(int sw_y=0; sw_y < nupl[l]; sw_y += grid.y*BLOCK_SIDE) {
					MMMulDevBack<<< grid,block,0,streams[str]>>>(d_h2h +sw_y, d_w +sw_x+sw_y*nupl[l+1], d_delta +sw_x, d_dest_delta + sw_y, d_delta_weight_dest + str*nupl[l]*nupl[l+1] +sw_x+sw_y*nupl[l+1], d_delta_bias_dest+ str*nupl[l+1] +sw_x, nupl[l], nupl[l+1], min(nupl[l]-sw_y,grid.y*BLOCK_SIDE) ,min(nupl[l+1]-sw_x,grid.x*BLOCK_SIDE),(1-sw_y),l, strsz);
				}
			}		
		}
		optimum_grid_x(&grid, OPTIMUM_BLOCK_NUM, nupl[0]/BLOCK_SIDE, nupl[1]);

		d_h2h = H2H + offset*nupl[0];
		d_w = WeightH2H;
		d_delta_weight_dest = TempDeltaWeightH2H;
		d_delta_bias_dest = TempDeltaBiasH2H;
		d_delta = Delta + offset*nupl[1];

		for(int sw_x=0; sw_x < nupl[1]; sw_x += grid.x*BLOCK_SIDE){
			for(int sw_y=0; sw_y < nupl[0]; sw_y += grid.y*BLOCK_SIDE) {
				MMMulDevBack<<< grid,block,0,streams[str]>>>(d_h2h +sw_y, d_w +sw_x+sw_y*nupl[1], d_delta +sw_x, NULL, d_delta_weight_dest + str*nupl[0]*nupl[1] +sw_x+sw_y*nupl[1], d_delta_bias_dest+ str*nupl[1] +sw_x, nupl[0], nupl[1], min(nupl[0]-sw_y,grid.y*BLOCK_SIDE) ,min(nupl[1]-sw_x,grid.x*BLOCK_SIDE),(1-sw_y),0, strsz);
			}
		}
  }
  /*  end BACKPROPAGATION */
	/* REDUCTION (SOLO UNO STREAM)*/
	for (int l = (layers -2); l >= 0; l--) {
		optimum_grid_x(&grid, OPTIMUM_BLOCK_NUM, nupl[l]/BLOCK_SIDE, nupl[l + 1]);

		d_w = WeightH2H + H_matrix_W_index[l];
		d_bias = BiasH2H + H_matrix_B_index[l];
		d_delta_weight = DeltaWeightH2H + H_matrix_W_index[l];
		d_delta_bias = DeltaBiasH2H + H_matrix_B_index[l];
		d_delta_weight_dest = TempDeltaWeightH2H + NSTREAMS*H_matrix_W_index[l];
    d_delta_bias_dest = TempDeltaBiasH2H + NSTREAMS*H_matrix_B_index[l];
    
    /* la dimensione della griglia è fissata, per tanto bisogna "spostarla" lungo la matrice deltaW */
		for(int sw_x=0; sw_x < nupl[l+1]; sw_x += grid.x*BLOCK_SIDE){
			for(int sw_y=0; sw_y < nupl[l];sw_y += grid.y*BLOCK_SIDE) {
				MMMulReduction<<<grid,block>>>(d_w +sw_x+sw_y*nupl[l+1], d_bias+ sw_x, d_delta_weight +sw_x+sw_y*nupl[l+1], d_delta_bias +sw_x , d_delta_weight_dest, d_delta_bias_dest, sw_x+sw_y*nupl[l+1], sw_x,  nupl[l], nupl[l+1], min(nupl[l]-sw_y,grid.y*BLOCK_SIDE) ,min(nupl[l+1]-sw_x,grid.x*BLOCK_SIDE),(1-sw_y),NSTREAMS);
			}
		}
  }
  /* END REDUCTION */

}