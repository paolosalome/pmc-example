#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATA float
#define BOOL int
#define MAX_ERR 1e-2

//Grid features

#define OPTIMUM_BLOCK_NUM 4 
#define BLOCK_SIDE	16 

#define OPTIMUM_BLOCK_NUM_FIRST_LAYER 2
#define BLOCK_SIDE_FIRST_LAYER 32

/*Struct Grid Settings*/

typedef struct grid_settings {
	unsigned int grid[3];
	unsigned int block[3];
}grid_settings;

grid_settings gs = { { OPTIMUM_BLOCK_NUM_FIRST_LAYER, OPTIMUM_BLOCK_NUM, OPTIMUM_BLOCK_NUM },{ BLOCK_SIDE_FIRST_LAYER,BLOCK_SIDE,BLOCK_SIDE } };


/* struct host_to_dev_mem {
	DATA WeightH2H[GLOBAL_W_SIZE];
	DATA BiasH2H[GLOBAL_BIAS_SIZE];
	DATA DeltaWeightH2H[GLOBAL_W_SIZE];
	DATA DeltaBiasH2H[GLOBAL_BIAS_SIZE];
	DATA Delta[GLOBAL_DELTA_SIZE];
	DATA H2H[GLOBAL_H_SIZE];
	int matrix_W_index[TOTAL_LAYER - 1];//INDEX for padding in Weight 
	int matrix_B_index[TOTAL_LAYER - 1];//INDEX for padding in Bias 
	int matrix_DELTA_index[TOTAL_LAYER - 1];//INDEX for padding in Delta
	int matrix_H2H_index[TOTAL_LAYER];//INDEX for padding in H2H
} ;

struct dev_struct {
	DATA WeightH2H[GLOBAL_W_SIZE];
	DATA BiasH2H[GLOBAL_BIAS_SIZE];
	DATA DeltaWeightH2H[GLOBAL_W_SIZE];
	DATA DeltaBiasH2H[GLOBAL_BIAS_SIZE];
	DATA Delta[GLOBAL_DELTA_SIZE];
	DATA H2H[GLOBAL_H_SIZE];
	DATA TempDeltaWeightH2H[NSTREAMS*GLOBAL_W_SIZE];
	DATA TempDeltaBiasH2H[NSTREAMS*GLOBAL_BIAS_SIZE];
} ; */

//Texture reference (FOR TARGET MATRIX)
texture<DATA, 2, cudaReadModeElementType> texreference_target;

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

/*deviceReduceBlockAtomicKernel*/
__inline__ __device__ DATA warpReduceSum(DATA);
__inline__ __device__ DATA blockReduceSum(DATA);
__global__ void deviceReduceBlockAtomicKernel(DATA *, DATA*, int);

/*MMMul(for feedforward)*/
__device__ void MMMulDevPartialFeed(DATA *, DATA *, DATA *, DATA *, DATA*, DATA *, unsigned int, unsigned int, unsigned int, unsigned int, int);
__global__ void MMMulDevFeed(DATA *, DATA *, DATA *, DATA *, DATA *, DATA*, unsigned int, unsigned int, unsigned int, unsigned int, int, int);

/* MMMul backprog*/
__device__ void MMMulDevPartialBack(DATA* , DATA* , DATA* , DATA* , DATA* , DATA* , int , int , int , int , BOOL, int, int, DATA);
__global__ void MMMulDevBack(DATA* , DATA* , DATA* , DATA* ,DATA* , DATA* , int , int , int , int , BOOL, int, int, DATA );

__device__ void MMMulReductionBlock(DATA* , DATA* , DATA* , DATA* , DATA* , DATA* , int , int , int , int ,  int , int , BOOL, int, DATA );
__global__ void MMMulReduction(DATA* , DATA* , DATA* , DATA* , DATA* , DATA* , int , int , int , int ,  int , int , BOOL, int, DATA );


/*HOST*/
void feedforward(DATA *, DATA* , DATA* ,DATA*,DATA*, int* , int*, int* , int* ,
	DATA* , DATA* , DATA* , DATA* , DATA*, DATA*,
	DATA *, DATA *, int *, int , cudaStream_t *, BOOL,
	int  , int , int , int , int );
void backpropagation(DATA* , DATA* , DATA* , DATA* , DATA* , DATA* , int* , int*,  int* , int* ,
	DATA* , DATA* , DATA* , DATA* , DATA* ,DATA* ,DATA* , DATA* , 
	int*  , int , cudaStream_t* ,int  , int , int , int , int , int , DATA,DATA);

void HOST_feedforward(DATA *, DATA **, DATA **, DATA **, int *,int);
void printMat(DATA *, int, int);
void printErrorMat(DATA *, DATA*, int, int);
void MMMulHost(DATA *, DATA *, DATA *, DATA *, int, int, int);
void BackMMMulHost(DATA *, DATA *, DATA *, DATA *, DATA *, DATA *, DATA *, DATA *, int, int, int, int,DATA,DATA);

BOOL matsAreEquals(DATA *, DATA *, int, int);
DATA errorReductionHost(DATA *, int, int);

/*HOST ALLOCATION AND INITIALIZATION*/
void HOST_init_struct( DATA* , DATA* , DATA* , DATA* , DATA* , DATA* , int*, int*, int* , int* , int* , int , int );

/*----------------------------------------------------------------------MAIN---------------------------------------------------------------------------*/

int main(void) {

	int NEURO_INPUT=784; //#neurons of input layer
	int NEURO_H_0=56;	//#neurons of first hidden layer
	int NEURO_H_1=28;	//#neurons of second hidden layer
	int NEURO_OUTPUT = 10; //#neurons of output layer
	int TOTAL_PATT=	60000; //#total patterns
	int TOTAL_LAYER = 4; //#of layers
	DATA eta = 0.05f;
	DATA alpha = 0.8f;
	//Streams Settings
	int NSTREAMS = 3;
	int STREAMSIZE = TOTAL_PATT/NSTREAMS;

	int GLOBAL_H_SIZE = TOTAL_PATT * (NEURO_INPUT + NEURO_H_0 + NEURO_H_1 + NEURO_OUTPUT);
	int GLOBAL_DELTA_SIZE = TOTAL_PATT * (NEURO_H_0 + NEURO_H_1 +NEURO_OUTPUT);
	int GLOBAL_W_SIZE = (NEURO_INPUT*NEURO_H_0) + (NEURO_H_0*NEURO_H_1) + (NEURO_H_1*NEURO_OUTPUT);
	int GLOBAL_BIAS_SIZE = NEURO_H_0 + NEURO_H_1 + NEURO_OUTPUT;

	DATA *INPUT_MAT, *ERROR_MAT, *DEV_ERROR_MAT;
	DATA *ERROR, *DEV_ERROR;
	DATA *TARGET;
	cudaStream_t streams[NSTREAMS];

	int *nupl = (int*)malloc(TOTAL_LAYER * sizeof(int));

	/*++++------------------------------------ERRORS--------------------------------------------------++++*/

	ERROR_MAT = (DATA*)malloc(TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA)); // ERROR FOR CHECKING CORRECTNESS
	HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR_MAT, TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA))); //DEVICE ERROR MAT

	ERROR = (DATA*)malloc(sizeof(DATA)); // ERROR FOR CHECKING CORRECTNESS
	HANDLE_CUDA(cudaMalloc((void **)&DEV_ERROR, sizeof(DATA))); //DEVICE ERROR
	HANDLE_CUDA(cudaMemset(DEV_ERROR, 0, sizeof(DATA)));

	/*----------------------------------------ERRORS END--------------------------------------------------*/

	/*++++---------------------------init INPUT_MAT and TARGET (HOST)-----------------------------++++*/
	nupl[0] = NEURO_INPUT;
	nupl[1] = NEURO_H_0;
	nupl[2] = NEURO_H_1;
	nupl[TOTAL_LAYER - 1] = NEURO_OUTPUT;

	TARGET = (DATA*)malloc(NEURO_OUTPUT*TOTAL_PATT * sizeof(DATA)); //TARGET OF THE PATTERNS

	for (int i = 0; i < TOTAL_PATT; i++) {
		for (int j = 0; j < NEURO_OUTPUT; j++) {
			TARGET[i*NEURO_OUTPUT + j] = (DATA)rand() / (DATA)RAND_MAX;
		}
	}

	/*INPUT_MAT is pinned memory*/
	
	HANDLE_CUDA(cudaHostAlloc(&INPUT_MAT, NEURO_INPUT * TOTAL_PATT * sizeof(DATA), 0));
	//DATA r;
	for (int i = 0; i < TOTAL_PATT; i++) {
		for (int j = 0; j < NEURO_INPUT; j++) {
			//r= rand() / (DATA)RAND_MAX;
			INPUT_MAT[i*NEURO_INPUT + j] = (DATA)rand() / (DATA)RAND_MAX;
			//htdm->H2H[i*NEURO_INPUT+ j] = r;
		}
	}

	/*---------------------------end init INPUT_MAT and TARGET (HOST)-------------------------*/

	/*++++---------------------------data structures on host and device-------------------------++++*/
	/*	HOST MATRIX AND INDEXES FOR PADDING */
	
	DATA* H_WeightH2H = (DATA*)malloc(GLOBAL_W_SIZE*sizeof(DATA));
	DATA* H_BiasH2H = (DATA*)malloc(GLOBAL_BIAS_SIZE*sizeof(DATA));
	DATA* H_DeltaWeightH2H = (DATA*)calloc(GLOBAL_W_SIZE,sizeof(DATA));
	DATA* H_DeltaBiasH2H = (DATA*)calloc(GLOBAL_BIAS_SIZE,sizeof(DATA));
	DATA* H_Delta = (DATA*)malloc(GLOBAL_DELTA_SIZE*sizeof(DATA));
	DATA* H_H2H = (DATA*)malloc(GLOBAL_H_SIZE*sizeof(DATA));
	int H_matrix_W_index[TOTAL_LAYER - 1];//INDEX for padding in Weight 
	int H_matrix_B_index[TOTAL_LAYER - 1];//INDEX for padding in Bias 
	int H_matrix_DELTA_index[TOTAL_LAYER - 1];//INDEX for padding in Delta
	int H_matrix_H2H_index[TOTAL_LAYER];//INDEX for padding in H2H

	/*	DEVICE MATRIX  */
	DATA* WeightH2H, *BiasH2H, *DeltaWeightH2H, *DeltaBiasH2H, *Delta, *H2H, *TempDeltaWeightH2H, *TempDeltaBiasH2H;

	//Init weights and biases on host
	HOST_init_struct(H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_Delta, H_H2H,	H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index, nupl, TOTAL_LAYER, TOTAL_PATT);
	//Malloc the necessary space on device memory
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
	HANDLE_CUDA(cudaMallocArray(&DEV_TARGET_CUDA, &channel, NEURO_OUTPUT, TOTAL_PATT));
	HANDLE_CUDA(cudaMemcpyToArray(DEV_TARGET_CUDA, 0, 0, TARGET, NEURO_OUTPUT*TOTAL_PATT * sizeof(DATA), cudaMemcpyHostToDevice));

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

	/*++++-----------------------------------FEEDFORWARD-------------------------------------------++++*/

	cudaEvent_t start, stop;
	startTimer(&start, &stop);
	//feedforward(INPUT_MAT, htdm, dev_htdm, DEV_ERROR_MAT, DEV_ERROR, nupl, TOTAL_LAYER, streams, 1);
	BOOL first=1;
	feedforward(INPUT_MAT,H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index, /*HOST*/
		WeightH2H, BiasH2H, Delta, H2H, DeltaWeightH2H, DeltaBiasH2H, DEV_ERROR_MAT, DEV_ERROR,/*DEVICE*/
		nupl, TOTAL_LAYER, streams, first, GLOBAL_BIAS_SIZE , GLOBAL_W_SIZE, NSTREAMS, STREAMSIZE, TOTAL_PATT);/*VARIOUS*/

	stopAndPrint(&start, &stop);
	//cudaDeviceSynchronize();//
	
	HANDLE_CUDA(cudaMemcpy(ERROR, DEV_ERROR, sizeof(DATA), cudaMemcpyDeviceToHost));
	printf("Reduced Error: %f\n", *ERROR);	
	
	HANDLE_CUDA(cudaMemcpy(ERROR_MAT, DEV_ERROR_MAT, TOTAL_PATT*NEURO_OUTPUT * sizeof(DATA), cudaMemcpyDeviceToHost));
		//printMat(ERROR_MAT, TOTAL_PATT, NEURO_OUTPUT);
	DATA red_host = errorReductionHost(ERROR_MAT, TOTAL_PATT, NEURO_OUTPUT);
	printf("host reduction error : %f\n", red_host);
	/*-------------------------------------END---FEEDFORWARD-------------------------------------------*/
	/*++++-----------------------------------BACKPROPAGATION-------------------------------------------++++*/

	
	//backpropagation(htdm, dev_htdm , nupl, TOTAL_LAYER, streams);
	backpropagation(H_WeightH2H, H_BiasH2H, H_DeltaWeightH2H, H_DeltaBiasH2H, H_Delta, H_H2H, H_matrix_W_index, H_matrix_B_index, H_matrix_DELTA_index, H_matrix_H2H_index,
		WeightH2H, BiasH2H, DeltaWeightH2H, DeltaBiasH2H, Delta, H2H, TempDeltaWeightH2H, TempDeltaBiasH2H, 
		nupl, TOTAL_LAYER, streams, GLOBAL_BIAS_SIZE , GLOBAL_W_SIZE, GLOBAL_DELTA_SIZE, NSTREAMS, STREAMSIZE, TOTAL_PATT, eta, alpha);
	/*-------------------------------------END---BACKPROPAGATION-------------------------------------------*/



	/*++++--------------------------------deallocations------------------------------------++++*/
	//Host dealloc
	free(nupl);
	free(TARGET);
	free(ERROR_MAT);
	free(ERROR);
	cudaFree(DEV_ERROR_MAT);
	cudaFree(DEV_ERROR);
	cudaFreeHost(INPUT_MAT);
	//Unbinding texture
	cudaUnbindTexture(texreference_target);
	//Free cuda array
	cudaFreeArray(DEV_TARGET_CUDA);

	/*------------------------------------end deallocations------------------------------------*/

	return 0;
}


/*---------------------------------------------------------------------KERNEL--------------------------------------------------------------------------*/

/*DEVICE*/

/*++++---------------------------deviceReduceBlockAtomicKernel---------------------------++++*/

/*Warp reduction*/
__inline__ __device__ DATA warpReduceSum(DATA val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

/*Block reduction*/
__inline__ __device__ DATA blockReduceSum(DATA val) {

	static __shared__ DATA shared[32];
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

//Nella chiamata di questo kernel � meglio usare una griglia lineare di 8 blocchi con 256 threads ciascuno -- 
//In tal modo vengono limitati gli accessi alla shared memory rispetto all'implementazione con 2 blocchi da 1024 threads ciascuno
//Attenzione ai possibili arrotondamenti di numeri a virgola mobile dovuti alle atomicAdd.
__global__ void deviceReduceBlockAtomicKernel(DATA *in, DATA* out, int N) {
	DATA sum = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x ; i < N ; i += blockDim.x * gridDim.x) {
		sum += in[i];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0)
		atomicAdd(out, sum);
}

/*-------------------------------end--deviceReduceBlockAtomicKernel--------------------------*/

/*++++---------------------------MMMul--Feedforward-------------------------++++*/

/* h2h � il puntatore alla porzione dell'h2h globale da considerare in questa fase
(ad ogni passo il kernel che invoca questo device incrementa il puntatore h2h
in modo proporzionale al patt_per_step (e similmente h2h_dest) (vedi sotto)).
offset_y � la posizione considerata lungo le y (nelle matrici h2h, h2h_dest ed eventualmente error) durante la chiamata corrente a __device__.
Delta � calcolato per l'output layer (propagato poi con backpropagation) --> DeltaO[p][k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;
*/

__device__ void MMMulDevPartialFeed(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, DATA *delta, DATA *error, unsigned int row_w, unsigned int col_w, unsigned int num_pattern, unsigned int offset_y, int NEURO_OUTPUT) {

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

	DATA partial = 0.0f;
	int block_r_border = 0; // contatore che indica in che iterazione dei blocchi ci troviamo
	int current_inc;

	for (int wid = w_x, h2h_id = h2h_y; h2h_id <= end_h2h; wid += step_w, h2h_id += step_h2h) {

		block_r_border += block_dim;

		//__shared__ DATA shared_w[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER+1]; Non possiamo ancora giustificare il miglioramento nei tempi.
		__shared__ DATA shared_w[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];
		__shared__ DATA shared_h2h[BLOCK_SIDE_FIRST_LAYER][BLOCK_SIDE_FIRST_LAYER];

		int t_index_w = wid + tx + ty*col_w;
		int t_index_h2h = h2h_id + tx + ty*row_w;

		//Attenzione alla divergenza dei threads (vedi CCC pag.137)
		shared_h2h[ty][tx] = (t_index_h2h < num_pattern*row_w) ? (h2h[t_index_h2h]) : (0.0f);
		shared_w[ty][tx] = (t_index_w < col_w*row_w) ? (w[t_index_w]) : (0.0f);

		__syncthreads();

		current_inc = row_w - (block_r_border - block_dim);

		min = (current_inc < block_dim) ? (current_inc) : (block_dim);

		#pragma unroll(2)
		for (int k = 0; k < min; k++) {
			partial += shared_h2h[ty][k] * shared_w[k][tx];
		}

		__syncthreads();
	}

	//Attenzione alla divergenza dei threads (vedi CCC pag.137)
	if (dest_x < col_w && dest_y < num_pattern) {

		DATA out = (DATA)1.0 / (DATA)(1.0 + exp(-(partial + biases[dest_x])));
		h2h_dest[dest_y*col_w + dest_x] = out; //SIGMA

		//Se siamo nell'ultimo passo
		if (col_w == NEURO_OUTPUT) {
			
			DATA target = tex2D(texreference_target, dest_x, dest_y + offset_y);

			//Scrivi nella posizione corrispondente della matrice di ERRORE
			/*0.5*(Target[p][k] - Output[p][k])*(Target[p][k] - Output[p][k])*/
			error[dest_y*col_w + dest_x] = 0.5*(target - out)*(target - out);

			//Scrivi nella posizione corrispondente della matrice DELTA
			/*(Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k])*/
			delta[dest_y*col_w + dest_x] = (target - out)*(out)*(1 - out);
		}
	}
}

/*patt_per_step � il numero di pattern (quando possibile...) da considerare in ciascuna iterazione su h2h*/
/*Questo kernel ad ogni passo incrementa il puntatore ad h2h di num_patt_per_step*NEURO_L_L_1 (e similmente h2h_dest),
controlla che sia ancora nel range di h2h, e calcola num_pattern (vedi sopra) in funzione dei
pattern mancanti.
stream_offset_y � la posizione lungo le y da cui parte (nelle matrici h2h e h2h_dest) lo stream corrente.
*/
//Dove ora c'� STREAMSIZE prima c'era TOTAL_PATT
__global__ void MMMulDevFeed(DATA *h2h, DATA *w, DATA *biases, DATA *h2h_dest, DATA *delta, DATA *error, unsigned int row_w, unsigned int col_w, unsigned int patt_per_step, unsigned int stream_offset_y,
						int STREAMSIZE, int NEURO_OUTPUT) {

	unsigned int current_patts;
	unsigned int remaining_patts;
	const int pos_block_y = blockIdx.y*blockDim.x; //Posizione del blocco corrente rispetto alla griglia lungo le y
												   //Assumiamo che i blocchi siano quadrati (blockDim.x = blockDim.y)		

	for (unsigned int y = 0; y < STREAMSIZE; y += patt_per_step) {

		remaining_patts = STREAMSIZE - y;
		current_patts = (remaining_patts < patt_per_step) ? (remaining_patts) : (patt_per_step);

		if (pos_block_y >= current_patts) { return; }

		MMMulDevPartialFeed(h2h + y*row_w, w, biases, h2h_dest + y*col_w, delta + y*NEURO_OUTPUT, error + y*NEURO_OUTPUT, row_w, col_w, current_patts, stream_offset_y + y, NEURO_OUTPUT);
	}
}

/*-------------------------------end--MMMul--Feedforward------------------------*/

/* la matrice di destinazione è width_h2h x width_delta     */
/* h2h_corner,delta_corner  sono in previsione di una "sliding grid" */
__device__ void MMMulDevPartialBack(DATA* h2h, DATA* w, DATA* delta, DATA* dest_delta, DATA* delta_weight_dest, DATA* delta_bias_dest, int width_h2h, int width_delta, int h2h_right_limit, int delta_right_limit, BOOL enable_bias, int layer, int STREAMSIZE, DATA eta){
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    /* int idx = t_x + blockIdx.x*BLOCK_SIDE ; 
    int idy = t_y + blockIdx.y*BLOCK_SIDE ;  */
    int pattern;
    int h2h_corner = blockIdx.y*BLOCK_SIDE;
    int delta_corner = blockIdx.x*BLOCK_SIDE;

    __shared__ DATA temp_shifted_mul[BLOCK_SIDE][BLOCK_SIDE*BLOCK_SIDE];//può contenere diversi 0 nei casi sui bordi
    __shared__ DATA temp_sum_delta_h2h[BLOCK_SIDE*BLOCK_SIDE];//può essere riciclato per W
    __shared__ DATA block_h2h[BLOCK_SIDE*BLOCK_SIDE];
    __shared__ DATA block_w[BLOCK_SIDE*(BLOCK_SIDE+1)];//usefull for avoid bank conflict
    __shared__ DATA block_delta[BLOCK_SIDE*BLOCK_SIDE];
    __shared__ DATA bias_to_update[BLOCK_SIDE*BLOCK_SIDE];

    int max_b_x = ((delta_corner + BLOCK_SIDE) < delta_right_limit) ? BLOCK_SIDE: (delta_right_limit - delta_corner);
    int max_a_x = ((h2h_corner + BLOCK_SIDE) < h2h_right_limit) ? BLOCK_SIDE: (h2h_right_limit - h2h_corner);

	temp_sum_delta_h2h[t_x+t_y*BLOCK_SIDE]=0.0f;
	if(layer>0)
	    block_w[t_x*BLOCK_SIDE+t_y] = (max_a_x > t_y && max_b_x > t_x) ? w[t_y*width_delta + t_x]:0.0f;

    if(enable_bias==1)
        bias_to_update[t_y*BLOCK_SIDE + t_x] = 0.0f;

    for(int curr_patterns=0;curr_patterns<STREAMSIZE;curr_patterns+=BLOCK_SIDE){
        pattern = (curr_patterns  + BLOCK_SIDE > STREAMSIZE) ? (STREAMSIZE-curr_patterns): BLOCK_SIDE ;

        block_h2h[t_y*BLOCK_SIDE+t_x]= ((curr_patterns+ t_y) < STREAMSIZE && max_a_x > t_x) ? h2h[t_y*width_h2h + t_x +curr_patterns*width_h2h]:0.0f;
		block_delta[t_y*BLOCK_SIDE+t_x] = ((curr_patterns+ t_y) < STREAMSIZE && max_b_x > t_x) ? delta[t_y*width_delta + t_x + curr_patterns*width_delta]:0.0f;
        __syncthreads();
        //DATA val = ((curr_patterns+ t_y) < P && max_b_x > t_x) ? delta[t_y*width_delta + t_x + curr_patterns*width_delta]:0.0f;
        DATA val = block_delta[t_y*BLOCK_SIDE+t_x];
        DATA temp=0.0f;
        
        for(int i=0 ;i<BLOCK_SIDE;i++){
			if(layer>0)
            	temp += block_delta[t_y*BLOCK_SIDE+i]*block_w[i*BLOCK_SIDE+t_x];//product delta*W by trd[ty][tx]
            temp_shifted_mul[t_y][ t_x + i*BLOCK_SIDE ] =  val*block_h2h[t_y*BLOCK_SIDE+i];
        }    

        __syncthreads();
        if(layer > 0 && t_y < pattern)
            atomicAdd(&dest_delta[t_y*width_h2h+ curr_patterns*width_h2h + t_x], temp*block_h2h[t_y*BLOCK_SIDE+t_x]*(1-block_h2h[t_y*BLOCK_SIDE+t_x]));//product 
        if(t_y==0){
            for(int j=t_x,index=0; index<BLOCK_SIDE;j+=BLOCK_SIDE, index++ ){
                for(int i=0 ;i<pattern;i++){
                    temp_sum_delta_h2h[j] += eta*temp_shifted_mul[i][j];
                }
            }
        }
        __syncthreads();
        if(enable_bias==1)//solo i blocchi con blocky = 0
            bias_to_update[t_y*BLOCK_SIDE + t_x] += block_delta[t_y*BLOCK_SIDE + t_x];
    }
    if( (t_y + h2h_corner) < h2h_right_limit && (t_x + delta_corner) < delta_right_limit){
        delta_weight_dest[t_x+t_y*width_delta] = temp_sum_delta_h2h[t_y*BLOCK_SIDE+ t_x];
    }
    if(enable_bias==1 &&  t_y==0 && (t_x + delta_corner) < delta_right_limit){
        DATA tempBias=0.0f;
        for(int i=0;i<BLOCK_SIDE;i++)
            tempBias+=bias_to_update[i*BLOCK_SIDE+t_x];
        delta_bias_dest[t_x] = eta*tempBias ;
    }
    //__syncthreads();
}

/* si può la riduzione finale di W sommando i delta calcolati e riaggiornare quindi il delta W con gli stessi .
 oppure si fa prima ma bisogna salvarlo a parte e non sovrascrivere subito la matrice di partenza (problemi di concorrenza con altri stream) 
 la matrice di desinazione avrà streams*L*(L+1) elementi . si effettua la riduzione sullo stream principale per salvarla su quella giusta
 */


/* THR_DEST has nupl[L]*nupl[l+1] element*/



__device__ void MMMulReductionBlock(DATA* W, DATA* BIAS, DATA* DELTA_WEIGHT, DATA* DELTA_BIAS, DATA* DELTA_WEIGHT_DEST, DATA* DELTA_BIAS_DEST, int offset_weight, int offset_bias, int width_h2h, int width_delta,  int Y_right_limit, int X_right_limit, BOOL enable_bias, int NSTREAMS, DATA alpha){
    
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int b_x = blockIdx.x*BLOCK_SIDE;
    int b_y = blockIdx.y*BLOCK_SIDE;
    int offset_block_w = b_x+b_y*width_delta;
    int offset_block_bias = b_x;
 
    if( (b_x + t_x) < X_right_limit && (b_y+ t_y) < Y_right_limit){
       
        DATA dw_loc = alpha*DELTA_WEIGHT[t_x+t_y*width_delta];
        DATA dbias_loc;
                
        for(int i=0 ;i<NSTREAMS; i++){
            dw_loc += DELTA_WEIGHT_DEST[i*width_delta*width_h2h+ offset_weight + offset_block_w + t_x + t_y*width_delta];//offset della griglia + offset blocco + offset thr
            if(enable_bias==1 && t_y==0){
                dbias_loc += DELTA_BIAS_DEST[i*width_delta+ offset_bias + offset_block_bias + t_x];
                //printf("enable_bias [%d][%d]< %d >-- local %f, stream %d, get by stream %f\n",b_y+t_y,b_x+t_x,enable_bias, dbias_loc, i,DELTA_BIAS_DEST[i*width_delta+ offset_bias + offset_block_bias + t_x]);
            }
        }
        if(enable_bias==1 && t_y==0){
            //printf("enable_bias [%d][%d] -- DELTA BIAS:%f    local %f,  %f\n",b_y+t_y,b_x+t_x, DELTA_BIAS[t_x]+dbias_loc ,dbias_loc, alpha*DELTA_BIAS[t_x]);
            BIAS[t_x] += dbias_loc + alpha*DELTA_BIAS[t_x];
            DELTA_BIAS[t_x] = dbias_loc + alpha*DELTA_BIAS[t_x];
        }
        W[t_x + t_y*width_delta] += dw_loc;
        DELTA_WEIGHT[t_x + t_y*width_delta] = dw_loc;
        //printf("enable_bias [%d][%d] -- DELTA WEIGHT:%f    local %f  %d-%d\n",b_y+t_y,b_x+t_x, DELTA_WEIGHT[t_x + t_y*width_delta],dw_loc,Y_right_limit,X_right_limit);
    }
}

__global__ void MMMulReduction(DATA* W, DATA* BIAS, DATA* DELTA_WEIGHT, DATA* DELTA_BIAS, DATA* DELTA_WEIGHT_DEST, DATA* DELTA_BIAS_DEST, int offset_weight, int offset_bias, int width_h2h, int width_delta,  int Y_right_limit, int X_right_limit, BOOL enable_bias, int NSTREAMS, DATA alpha){
    int b_x = blockIdx.x*BLOCK_SIDE;
    int b_y = blockIdx.y*BLOCK_SIDE;
    //enable bias vale 1 se la griglia si è spostata lungo la x . Gli unici blocchi che calcoleranno il delta bias sono quelli con blockIdy = 0
    if(b_x < X_right_limit && b_y <Y_right_limit)
        MMMulReductionBlock(W+b_x+b_y*width_delta, BIAS+ b_x, DELTA_WEIGHT+b_x+b_y*width_delta, DELTA_BIAS+ b_x, DELTA_WEIGHT_DEST, DELTA_BIAS_DEST, offset_weight, offset_bias, width_h2h, width_delta,  Y_right_limit, X_right_limit, enable_bias*(1-blockIdx.y),NSTREAMS,alpha);
    //__syncthreads();
}
__global__ void MMMulDevBack(DATA* H2H, DATA* W, DATA* DELTA, DATA* DEST_DELTA, DATA* DELTA_WEIGHT_DEST, DATA* DELTA_BIAS_DEST, int width_h2h, int width_delta, int h2h_right_limit, int delta_right_limit, BOOL enable_bias,int layer, int STREAMSIZE, DATA eta){
    int b_x = blockIdx.x*BLOCK_SIDE;
    int b_y = blockIdx.y*BLOCK_SIDE;
    //enable bias vale 1 se la griglia si è spostata lungo la x . Gli unici blocchi che calcoleranno il delta bias sono quelli con blockIdy = 0
	
	if(b_x < delta_right_limit && b_y <h2h_right_limit)
        MMMulDevPartialBack(H2H +b_y, W +b_x+b_y*width_delta, DELTA +b_x, DEST_DELTA+b_y, DELTA_WEIGHT_DEST +b_x+b_y*width_delta, DELTA_BIAS_DEST +b_x, width_h2h, width_delta, h2h_right_limit, delta_right_limit, enable_bias*(1-blockIdx.y),layer, STREAMSIZE, eta);
    //__syncthreads();
}




/*HOST*/

/*FIRT PHASE OF THE ALGORITHM -- THE INPUT IS TRANSMITTED VIA THE NETWORK*/ 
// OCCHIO NON SERVE PASSARE LE MATRICI H_H2H, H_DELTA_H2H, H_DELTA_WH2H, H_DELTA_BIASH2H, DELTA_WH2H,DELTA_BIAS
void feedforward (DATA *INPUT, 
	DATA* H_WeightH2H, DATA* H_BiasH2H, DATA* H_DeltaWeightH2H, DATA* H_DeltaBiasH2H, int* H_matrix_W_index, int* H_matrix_B_index, int* H_matrix_DELTA_index, int* H_matrix_H2H_index,
	DATA* WeightH2H, DATA* BiasH2H, DATA* Delta, DATA* H2H, DATA* DeltaWeightH2H, DATA* DeltaBiasH2H,
	DATA *dev_error_mat, DATA *dev_error, int *nupl, int layers, cudaStream_t *streams, BOOL first_epoch,
	int GLOBAL_BIAS_SIZE , int GLOBAL_W_SIZE, int NSTREAMS, int STREAMSIZE, int TOTAL_PATT) {
	//cudaEvent_t start, stop;

	//Grid setting
	dim3 grid, block;
	unsigned int patt_per_step;
	//Useful pointers
	DATA *h2h, *w, *bias, *h2h_dest, *delta, *error;

	//offset
	int offset;

	//startTimer(&start, &stop);
	if (first_epoch) {//il fattore 2 tiene conto anche dei delta utili sucessivamente nella fase di backpropagation
		//HANDLE_CUDA(cudaMemcpy(dev_htdm, htdm, 2*(GLOBAL_BIAS_SIZE + GLOBAL_W_SIZE) * sizeof(DATA), cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(WeightH2H, H_WeightH2H, GLOBAL_W_SIZE * sizeof(DATA), cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(BiasH2H, H_BiasH2H, GLOBAL_BIAS_SIZE * sizeof(DATA), cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(DeltaWeightH2H, H_DeltaWeightH2H, GLOBAL_W_SIZE * sizeof(DATA), cudaMemcpyHostToDevice));
		HANDLE_CUDA(cudaMemcpy(DeltaBiasH2H, H_DeltaBiasH2H, GLOBAL_BIAS_SIZE * sizeof(DATA), cudaMemcpyHostToDevice));


	}
	//stopAndPrint(&start, &stop);

	for (int i = 0; i < NSTREAMS; i++) {

		block.x = gs.block[0];
		block.y = gs.block[0];
		grid.x = (nupl[1] + block.x - 1) / block.x;
		grid.y = gs.grid[0] / grid.x;

		patt_per_step = grid.y * block.y;

		offset = i*STREAMSIZE;
		//Set pointers
		h2h = H2H + offset*nupl[0];
		w = WeightH2H;
		bias = BiasH2H;
		h2h_dest = H2H + H_matrix_H2H_index[1] + offset*nupl[1];
		delta = Delta + H_matrix_DELTA_index[layers - 2] + offset*nupl[layers - 1];
		error = dev_error_mat + offset*nupl[layers - 1];
		//Pointers set up

		if (first_epoch) {
			HANDLE_CUDA(cudaMemcpyAsync(h2h, INPUT + offset*nupl[0], nupl[0] * STREAMSIZE * sizeof(DATA), cudaMemcpyHostToDevice, streams[i]));
		}

		MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, error, nupl[0], nupl[1], patt_per_step, offset, STREAMSIZE, nupl[layers-1]);

		for (int l = 1; l < (layers - 1); l++) {

			block.x = gs.block[l];
			block.y = gs.block[l];
			grid.x = (nupl[l + 1] + block.x - 1) / block.x;
			grid.y = gs.grid[l] / grid.x;

			patt_per_step = grid.y * block.y;
			//Set pointers
			h2h = H2H + H_matrix_H2H_index[l] + offset*nupl[l];
			w = WeightH2H + H_matrix_W_index[l];
			bias = BiasH2H + H_matrix_B_index[l];
			h2h_dest = H2H + H_matrix_H2H_index[l + 1] + offset*nupl[l + 1];
			//Delta and error already set up
			//Pointers set up

			MMMulDevFeed << <grid, block, 0, streams[i] >> > (h2h, w, bias, h2h_dest, delta, error, nupl[l], nupl[l + 1], patt_per_step, offset, STREAMSIZE, nupl[layers-1]);
		}
	}
	//**HERE**
	//Error reduction (default stream)
	deviceReduceBlockAtomicKernel << <OPTIMUM_BLOCK_NUM * 2, BLOCK_SIDE*BLOCK_SIDE >> > (dev_error_mat, dev_error, TOTAL_PATT*nupl[layers-1]);
}
/*	 BACKPROPAGATION	*/
void backpropagation(DATA* H_WeightH2H, DATA* H_BiasH2H, DATA* H_DeltaWeightH2H, DATA* H_DeltaBiasH2H, DATA* H_Delta, DATA* H_H2H, int* H_matrix_W_index, int * H_matrix_B_index,int* H_matrix_DELTA_index, int* H_matrix_H2H_index,
	DATA* WeightH2H, DATA* BiasH2H, DATA* DeltaWeightH2H, DATA* DeltaBiasH2H, DATA* Delta,DATA* H2H,DATA* TempDeltaWeightH2H, DATA* TempDeltaBiasH2H, 
	int* nupl , int layers, cudaStream_t* streams,int GLOBAL_BIAS_SIZE , int GLOBAL_W_SIZE, int GLOBAL_DELTA_SIZE, int NSTREAMS, int STREAMSIZE, int TOTAL_PATT, DATA eta, DATA alpha){
    dim3 grid,block;
    block.x= BLOCK_SIDE;
    block.y= BLOCK_SIDE;

	DATA *d_h2h, *d_w, *d_bias, *d_delta_weight, *d_delta_bias, *d_delta, *d_dest_delta, *d_delta_weight_dest, *d_delta_bias_dest;
	
	int offset;
	for(int str=0;str<NSTREAMS;str++){
		offset=str*STREAMSIZE;

		for (int l = (layers -2); l > 0; l--) {
			optimum_grid_x(&grid, OPTIMUM_BLOCK_NUM, nupl[l]/BLOCK_SIDE, nupl[l + 1]);
			//printf("grid :%d %d\n",grid.y,grid.x);
			//Set pointers
			d_h2h = H2H + H_matrix_H2H_index[l] + offset*nupl[l];
			d_w = WeightH2H + H_matrix_W_index[l];
			d_delta_weight_dest = TempDeltaWeightH2H + NSTREAMS*H_matrix_W_index[l];
			d_delta_bias_dest = TempDeltaBiasH2H + NSTREAMS*H_matrix_B_index[l];
			d_delta = Delta + H_matrix_DELTA_index[l] + offset*nupl[l+1];
			d_dest_delta = Delta + H_matrix_DELTA_index[l-1] + offset*nupl[l];
			
			for(int sw_x=0; sw_x < nupl[l+1]; sw_x += grid.x*BLOCK_SIDE){
				for(int sw_y=0; sw_y < nupl[l]; sw_y += grid.y*BLOCK_SIDE) {
					MMMulDevBack<<< grid,block,0,streams[str]>>>(d_h2h +sw_y, d_w +sw_x+sw_y*nupl[l+1], d_delta +sw_x, d_dest_delta + sw_y, d_delta_weight_dest + str*nupl[l]*nupl[l+1] +sw_x+sw_y*nupl[l+1], d_delta_bias_dest+ str*nupl[l+1] +sw_x, nupl[l], nupl[l+1], min(nupl[l]-sw_y,grid.y*BLOCK_SIDE) ,min(nupl[l+1]-sw_x,grid.x*BLOCK_SIDE),(1-sw_y),l, STREAMSIZE,eta);
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
				MMMulDevBack<<< grid,block,0,streams[str]>>>(d_h2h +sw_y, d_w +sw_x+sw_y*nupl[1], d_delta +sw_x, NULL, d_delta_weight_dest + str*nupl[0]*nupl[1] +sw_x+sw_y*nupl[1], d_delta_bias_dest+ str*nupl[1] +sw_x, nupl[0], nupl[1], min(nupl[0]-sw_y,grid.y*BLOCK_SIDE) ,min(nupl[1]-sw_x,grid.x*BLOCK_SIDE),(1-sw_y),0, STREAMSIZE,eta);
			}
		}
	}
	/* REDUCTION */
	for (int l = (layers -2); l >= 0; l--) {
		optimum_grid_x(&grid, OPTIMUM_BLOCK_NUM, nupl[l]/BLOCK_SIDE, nupl[l + 1]);

		d_w = WeightH2H + H_matrix_W_index[l];
		d_bias = BiasH2H + H_matrix_B_index[l];
		d_delta_weight = DeltaWeightH2H + H_matrix_W_index[l];
		d_delta_bias = DeltaBiasH2H + H_matrix_B_index[l];
		d_delta_weight_dest = TempDeltaWeightH2H + NSTREAMS*H_matrix_W_index[l];
		d_delta_bias_dest = TempDeltaBiasH2H + NSTREAMS*H_matrix_B_index[l];

		for(int sw_x=0; sw_x < nupl[l+1]; sw_x += grid.x*BLOCK_SIDE){
			for(int sw_y=0; sw_y < nupl[l];sw_y += grid.y*BLOCK_SIDE) {
				MMMulReduction<<<grid,block>>>(d_w +sw_x+sw_y*nupl[l+1], d_bias+ sw_x, d_delta_weight +sw_x+sw_y*nupl[l+1], d_delta_bias +sw_x , d_delta_weight_dest, d_delta_bias_dest, sw_x+sw_y*nupl[l+1], sw_x,  nupl[l], nupl[l+1], min(nupl[l]-sw_y,grid.y*BLOCK_SIDE) ,min(nupl[l+1]-sw_x,grid.x*BLOCK_SIDE),(1-sw_y),NSTREAMS,alpha);
			}
		}
	}


	/*----check error------*/

	HANDLE_CUDA(cudaMemcpy( H_Delta, Delta, (GLOBAL_DELTA_SIZE - TOTAL_PATT*nupl[3] )*sizeof(DATA), cudaMemcpyDeviceToHost));
	HANDLE_CUDA(cudaMemcpy(H_H2H + TOTAL_PATT*nupl[0], H2H + TOTAL_PATT*nupl[0], TOTAL_PATT*(nupl[1]+nupl[2]+nupl[3])*sizeof(DATA), cudaMemcpyDeviceToHost));
	/*  delta_w,			delta_b e 		dest_delta 		HOST
		new_delta_weight	new_delta_bias		new_delta		DEVICE
	*/
	DATA *delta_w,*delta_b,*new_delta_weight,*new_delta_bias,*old_delta,*new_delta,*dest_delta;
	
	DATA *h2h,*w, /* *bias,*/*delta_weight,*delta_bias; //questi si prendono direttamente dalla struct htdm
	for (int l = (layers -2); l >= 0; l--) {
		delta_w=(DATA*)calloc(nupl[l]*nupl[l+1],sizeof(DATA));
		delta_b=(DATA*)calloc(nupl[l+1],sizeof(DATA));
		new_delta_weight = (DATA*)malloc(nupl[l]*nupl[l+1]*sizeof(DATA));
		new_delta_bias = (DATA*)malloc(nupl[l+1]*sizeof(DATA));
		old_delta= (DATA*)malloc(TOTAL_PATT*nupl[l+1]*sizeof(DATA));
		new_delta= (DATA*)malloc(TOTAL_PATT*nupl[l]*sizeof(DATA));
		dest_delta= (DATA*)malloc(TOTAL_PATT*nupl[l]*sizeof(DATA));

		h2h = H_H2H + H_matrix_H2H_index[l];
		w = H_WeightH2H + H_matrix_W_index[l];
		//bias = H_BiasH2H + H_matrix_B_index[l];
		delta_weight = H_DeltaWeightH2H + H_matrix_W_index[l];
		delta_bias = H_DeltaBiasH2H + H_matrix_B_index[l];
		
		d_dest_delta = (l>0)?Delta + H_matrix_DELTA_index[l-1]:NULL;
		d_delta = Delta + H_matrix_DELTA_index[l];

		d_delta_weight = DeltaWeightH2H + H_matrix_W_index[l];
		d_delta_bias = DeltaBiasH2H + H_matrix_B_index[l];
		
		HANDLE_CUDA(cudaMemcpy(old_delta,d_delta, TOTAL_PATT*nupl[l+1]*sizeof(DATA),cudaMemcpyDeviceToHost));
		if(l>0)
			HANDLE_CUDA(cudaMemcpy(new_delta,d_dest_delta, TOTAL_PATT*nupl[l]*sizeof(DATA),cudaMemcpyDeviceToHost));
		HANDLE_CUDA(cudaMemcpy(new_delta_bias,d_delta_bias, nupl[l+1]*sizeof(DATA),cudaMemcpyDeviceToHost));
		HANDLE_CUDA(cudaMemcpy(new_delta_weight,d_delta_weight, nupl[l]*nupl[l+1]*sizeof(DATA),cudaMemcpyDeviceToHost));

		BackMMMulHost(h2h, w, old_delta, delta_weight, delta_bias, delta_w, delta_b, dest_delta, TOTAL_PATT, nupl[l+1], nupl[l],l,eta,alpha);
		if(l>0){
			printf("\ndelta h2h  P*%d: \n",nupl[l]);
			matsAreEquals(new_delta,dest_delta,TOTAL_PATT,nupl[l]);
		}
		printf("delta W-h2h %d*%d: \n", nupl[l],nupl[l+1]);
		matsAreEquals(new_delta_weight,delta_w,nupl[l],nupl[l+1]);
/* 		if(l== (layers-2)) {
			printMat(delta_weight,nupl[l],nupl[l+1]);
			printMat(new_delta_weight,nupl[l],nupl[l+1]);
			printMat(delta_w,nupl[l],nupl[l+1]);
		} */

		printf("delta bias : 1*%d: \n",nupl[l+1]);
		matsAreEquals(new_delta_bias,delta_b,1,nupl[l+1]);
	}
	HANDLE_CUDA(cudaMemcpy( H_DeltaWeightH2H , DeltaWeightH2H, GLOBAL_W_SIZE*sizeof(DATA), cudaMemcpyDeviceToHost));
	HANDLE_CUDA(cudaMemcpy( H_DeltaBiasH2H, DeltaBiasH2H, (GLOBAL_BIAS_SIZE)*sizeof(DATA), cudaMemcpyDeviceToHost));
	/* -----end check error------*/
	
}




/*UTILITY FUNCTIONS*/

void HOST_feedforward(DATA *INPUT, DATA **W, DATA **BIAS, DATA **H2H, int *nupl,int TOTAL_PATT) {

	MMMulHost(INPUT, W[0], BIAS[0], H2H[0], TOTAL_PATT, nupl[0], nupl[1]);
	MMMulHost(H2H[0], W[1], BIAS[1], H2H[1], TOTAL_PATT, nupl[1], nupl[2]);
	MMMulHost(H2H[1], W[2], BIAS[2], H2H[2], TOTAL_PATT, nupl[2], nupl[3]);

}

/*Print a matrix*/
void printMat(DATA *mat, int rows, int cols) {

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

/*Print error matrix on host (for checking correctness of device)*/
void printErrorMat(DATA *TARGET, DATA *OUTPUT_MAT, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		printf("ROW %d : {", i);
		for (int j = 0; j < cols; j++) {
			printf("%f - ", 0.5*(TARGET[i*cols + j] - OUTPUT_MAT[i*cols + j])*(TARGET[i*cols + j] - OUTPUT_MAT[i*cols + j]));
		}
		printf("}");
		printf("\n\n");
	}
	printf("\n\n");
}

/*On host multiplication*/
void MMMulHost(DATA *H2H, DATA *W, DATA *BIAS, DATA *H2H_RES, int row_H2H, int col_H2H, int col_W) {

	for (int i = 0; i < row_H2H; i++) {
		for (int j = 0; j < col_W; j++) {
			DATA prod = 0.0;
			for (int k = 0; k < col_H2H; k++) {
				prod += H2H[i*col_H2H + k] * W[k*col_W + j];
			}
			H2H_RES[i*col_W + j] = (DATA)1.0 / (DATA)(1.0 + exp(-(prod + BIAS[j]))); // bias added
		}
	}
}
void BackMMMulHost(DATA *h2h, DATA * w, DATA * delta, DATA * delta_weight, DATA * delta_bias, DATA * new_delta_weight, DATA * new_delta_bias, DATA * dest_delta, int P,int width_delta,int width_h2h,int layer,DATA eta, DATA alpha){
	if(layer>0)
		for(int row=0;row<P;row++){
			for(int cola=0;cola<width_h2h;cola++){
				DATA temp= 0.0f;
				for(int colb=0;colb<width_delta;colb++)
					temp+= delta[row*width_delta+colb]*w[cola*width_delta+colb];    
				dest_delta[row*width_h2h+cola] = temp*h2h[row*width_h2h+cola]*(1.0f-h2h[row*width_h2h+cola]);
			}
		}
    for(int colb=0;colb<width_delta;colb++){
        new_delta_bias[colb] = alpha*delta_bias[colb];
        for(int cola=0;cola<width_h2h;cola++)
			new_delta_weight[cola*width_delta+colb] = alpha*delta_weight[cola*width_delta+ colb];
	}
    for(int row=0;row<P;row++){
        for(int colb=0;colb<width_delta;colb++){
            new_delta_bias[colb] += eta*delta[row*width_delta+colb] ;
            for(int cola=0;cola<width_h2h;cola++)
				new_delta_weight[cola*width_delta+colb]+= eta*h2h[row*width_h2h+cola]*delta[row*width_delta+colb];
        }
    }
}



/*Check device*/
BOOL matsAreEquals(DATA *A, DATA *B, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) { // the first column is for adapting the data
			float err = fabs(A[i*cols + j] - B[i*cols + j]);
			//printf("Error in i=%d,j=%d: %f\n", i, j, err);
			if (err >= MAX_ERR) { printf("row: %d, col: %d :: %f,%f\n", i, j, A[i*cols + j], B[i*cols + j]); return 0; }
		}
	}
	return 1;
}

/*Check device reduction*/
DATA errorReductionHost(DATA *error_mat, int rows, int cols) {

	DATA reduction = 0.0f;

	for (int i = 0; i < rows*cols; i++) {
		reduction += error_mat[i];
	}

	return reduction;
}

/*ALLOCATION FUNCTIONS*/

/*init struct on host*/
void HOST_init_struct( DATA* WeightH2H, DATA* BiasH2H, DATA* DeltaWeightH2H, DATA* DeltaBiasH2H, DATA* Delta, DATA* H2H,	int* matrix_W_index, int* matrix_B_index,  int* matrix_DELTA_index, int* matrix_H2H_index, int* nupl, int layers, int TOTAL_PATT) {
		
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

		matrix_H2H_index[layer] = nupl[layer - 1] * TOTAL_PATT + prev_sum[0];
		matrix_DELTA_index[layer] = nupl[layer] * TOTAL_PATT + prev_sum[1];
		matrix_W_index[layer] = nupl[layer - 1] * nupl[layer] + prev_sum[2];
		matrix_B_index[layer] = nupl[layer] + prev_sum[3];

		for (int i = 0; i < nupl[layer]; i++) {
			for (int j = 0; j < nupl[layer + 1]; j++) {
				WeightH2H[matrix_W_index[layer] + i*nupl[layer + 1] + j] = (DATA)rand() / (DATA)RAND_MAX;
				BiasH2H[matrix_B_index[layer] + j] = (DATA)rand() / (DATA)RAND_MAX;
				DeltaWeightH2H[matrix_W_index[layer] + i*nupl[layer + 1] + j] = (DATA)rand() / (DATA)RAND_MAX;
				DeltaBiasH2H[matrix_B_index[layer] + j] = (DATA)rand() / (DATA)RAND_MAX;
			}
		}

	}
	prev_sum[0] = matrix_H2H_index[layers - 2];
	matrix_H2H_index[layers - 1] = nupl[layers - 2] * TOTAL_PATT + prev_sum[0];

	for (int i = 0; i < nupl[0]; i++) {
		for (int j = 0; j < nupl[1]; j++) {
			WeightH2H[i*nupl[1] + j] = (DATA)rand() / (DATA)RAND_MAX;
			BiasH2H[j] = (DATA)rand() / (DATA)RAND_MAX;
		}
	}
}

//NON CANCELLARE !!! INSERIRE NEL FEEDFORWARD PER FARE TEST DI CORRETTEZZA NEL PUNTO **HERE**!!!
//RICORDARSI DI DECOMMENTARE LA 'r' NEL MAIN

/*
cudaDeviceSynchronize();
DATA **H2H_RES = (DATA**)malloc(TOTAL_LAYER * sizeof(DATA*));
for (int i = 0; i < TOTAL_LAYER; i++) {
H2H_RES[i] = (DATA*)malloc(TOTAL_PATT*nupl[i] * sizeof(DATA));
}
for (int l = 0; l < (layers - 1); l++) {

HANDLE_CUDA(cudaMemcpy(htdm->H2H+ htdm->matrix_H2H_index[l+1],dev_htdm->H2H + htdm->matrix_H2H_index[l+1], (TOTAL_PATT)* nupl[l+1] * sizeof(DATA), cudaMemcpyDeviceToHost));
MMMulHost( htdm->H2H + htdm->matrix_H2H_index[l], htdm->WeightH2H + htdm->matrix_W_index[l] , htdm->BiasH2H + htdm->matrix_B_index[l], H2H_RES[l + 1], TOTAL_PATT, nupl[l], nupl[l + 1]);

BOOL b = matsAreEquals(htdm->H2H+ htdm->matrix_H2H_index[l+1], H2H_RES[l + 1], TOTAL_PATT, nupl[l + 1]);
printf("layer%d %d\n",l, b);
}*/