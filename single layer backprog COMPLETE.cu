#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define min(a,b) (a) < (b) ? (a) : (b)
#define BOOL int
#define blockSide 16
#define blockNum 16
#define epsilon 1e-4
#define N 10
#define M 28
#define P 20000
#define DATA float


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


/* la matrice di destinazione è width_h2h x width_delta     */
/* h2h_corner,delta_corner  sono in previsione di una "sliding grid" */

__device__ void matrix_array_block(DATA* h2h, DATA* w, DATA* delta, DATA* thr_delta, DATA* dest_delta, int width_h2h, int width_delta, int A_right_limit, int B_right_limit){
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int idx = t_x + blockIdx.x*blockSide ; 
    int idy = t_y + blockIdx.y*blockSide ;
    int pattern;
    int h2h_corner = blockIdx.y*blockSide;
    int delta_corner = blockIdx.x*blockSide;

    __shared__ DATA temp_shifted_mul[blockSide][blockSide*blockSide];//può contenere diversi 0 nei casi sui bordi
    __shared__ DATA temp_sum_delta_h2h[blockSide*blockSide];//può essere riciclato per W
    __shared__ DATA block_h2h[blockSide*blockSide];
    __shared__ DATA block_w[blockSide*(blockSide+1)];//usefull for avoid bank conflict
    __shared__ DATA block_delta[blockSide*blockSide];

    int max_b_x = ((delta_corner + blockSide) < B_right_limit) ? blockSide: (B_right_limit - delta_corner);
    int max_a_x = ((h2h_corner + blockSide) < A_right_limit) ? blockSide: (A_right_limit - h2h_corner);

    temp_sum_delta_h2h[t_x+t_y*blockSide]=0.0f;
    block_w[t_x*blockSide+t_y] = (max_a_x > t_y && max_b_x > t_x) ? w[t_y*width_delta + t_x]:0.0f;

    for(int curr_patterns=0;curr_patterns<P;curr_patterns+=blockSide){
        pattern = (curr_patterns  + blockSide > P) ? (P-curr_patterns): blockSide ;

        DATA val = ((curr_patterns+ t_y) < P && max_b_x > t_x) ? delta[t_y*width_delta + t_x + curr_patterns*width_delta]:0.0f;

        block_h2h[t_y*blockSide+t_x]= ((curr_patterns+ t_y) < P && max_a_x > t_x) ? h2h[t_y*width_h2h + t_x +curr_patterns*width_h2h]:0.0f;
        block_delta[t_y*blockSide+t_x] = ((curr_patterns+ t_y) < P && max_b_x > t_x) ? delta[t_y*width_delta + t_x + curr_patterns*width_delta]:0.0f;
        __syncthreads();

        DATA temp=0.0f;
        for(int i=0 ;i<blockSide;i++){
        /*  QUI CI VA IL PRODOTTO TRA TEMP=W*DELTA   */
            temp += block_delta[t_y*blockSide+i]*block_w[i*blockSide+t_x];//product delta*W by trd[ty][tx]
            temp_shifted_mul[t_y][ t_x + i*blockSide ] =  val*block_h2h[t_y*blockSide+i];
           //temp_shifted_mul[t_y][ t_x + i*blockSide ] =  ( (curr_patterns+ t_y) < P && max_a_x > i) ? val*h2h[t_y*width_h2h + i +curr_patterns*width_h2h]:0.0f;
        }    

        __syncthreads();
        atomicAdd(&dest_delta[t_y*width_h2h+ curr_patterns*width_h2h + t_x], temp*block_h2h[t_y*blockSide+t_x]*(1-block_h2h[t_y*blockSide+t_x]));//product 

        if(t_y==0){
            for(int j=t_x,index=0; index<blockSide;j+=blockSide, index++ ){
                for(int i=0 ;i<pattern;i++)
                    temp_sum_delta_h2h[j] += temp_shifted_mul[i][j];
            }
        }
        __syncthreads();

    }
    if(t_y + h2h_corner < A_right_limit && t_x + delta_corner < B_right_limit)
        thr_delta[t_x+t_y*width_delta] = temp_sum_delta_h2h[t_y*blockSide+ t_x];
    
}
/* THR_DEST has nupl[L]*nupl[l+1] element*/
__global__ void matrix_mul(DATA* H2H, DATA* W, DATA* DELTA, DATA* THR_DELTA, DATA* DEST_DELTA, int width_h2h, int width_delta, int A_right_limit, int B_right_limit){
    int b_x = blockIdx.x*blockSide;
    int b_y = blockIdx.y*blockSide;
    if(b_x < B_right_limit && b_y <A_right_limit)
        matrix_array_block(H2H +b_y, W +b_x+b_y*width_delta, DELTA +b_x, THR_DELTA +b_x+b_y*width_delta, DEST_DELTA+b_y, width_h2h, width_delta, A_right_limit, B_right_limit);
    __syncthreads();
}
void optimum_grid_x(dim3* grid,int max_block,int y_limit){
    
    int x = min((N+blockSide-1)/blockSide,max_block);
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
void backward(DATA *host_h2h, DATA* host_delta, DATA* host_thread_delta, DATA* d_h2h, DATA* d_w, DATA* d_delta, DATA* d_thread_delta, DATA* d_dest_delta, int width_h2h, int width_delta){
    dim3 grid,block;
    optimum_grid_x(&grid,blockNum,width_h2h/blockSide);
    block.x= blockSide;
    block.y= blockSide;
    printf("grid :%d %d\n",grid.y,grid.x);
    cudaEvent_t start,stop;
    startTimer(&start,&stop);
    for(int sw_x=0; sw_x < width_delta; sw_x += grid.x*blockSide)
        for(int sw_y=0; sw_y < width_h2h;sw_y += grid.y*blockSide) {
            matrix_mul<<< grid,block >>>(d_h2h +sw_y, d_w +sw_x+sw_y*width_delta, d_delta +sw_x, d_thread_delta+sw_x+sw_y*width_delta, d_dest_delta + sw_y, width_h2h, width_delta, min(width_h2h-sw_y,grid.y*blockSide) ,min(width_delta-sw_x,grid.x*blockSide));
            //printf("grid :%d %d----limits:%d %d\n",sw_y,sw_x,min(width_h2h-sw_y,grid.y*blockSide),min(width_delta-sw_x,grid.x*blockSide));
        }
    stopAndPrint(&start,&stop);

}

/*Check device*/
BOOL matsAreEquals(DATA *A, DATA *B, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) { // the first column is for adapting the data
			float err = fabs(A[i*cols + j] - B[i*cols + j]);
			//printf("Error in i=%d,j=%d: %f\n", i, j, err);
			if (err >= epsilon) { printf("row: %d, col: %d\n", i, j); return 0; }
		}
	}
	return 1;
}

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

int main(){
    DATA *h2h, *w, *delta, *c_host, *dest_c, *new_delta, *delta_host;
    DATA *d_h2h, *d_w, *d_delta, *d_thread_delta, *d_dest_delta;

    h2h=(DATA *)malloc(P*M*sizeof(DATA));
    w=(DATA *)malloc(M*N*sizeof(DATA));
    delta=(DATA *)malloc(P*N*sizeof(DATA));
    new_delta=(DATA *)calloc(P*M,sizeof(DATA));
    delta_host=(DATA *)calloc(P*M,sizeof(DATA));
    c_host=(DATA *)calloc(M*N,sizeof(DATA));
    dest_c=(DATA *)calloc(M*N,sizeof(DATA));

    cudaMalloc((void**)&d_h2h,P*M*sizeof(DATA));
    cudaMalloc((void**)&d_w,M*N*sizeof(DATA));
    cudaMalloc((void**)&d_delta,P*N*sizeof(DATA));
    cudaMalloc((void**)&d_dest_delta,P*M*sizeof(DATA));
    cudaMalloc((void**)&d_thread_delta,M*N*sizeof(DATA));
/* -------------------------------init  -------------------*/
    for(int row=0;row<P;row++){
        for(int cola=0;cola<M;cola++)
            h2h[row*M+cola]=(DATA)rand() / (DATA)RAND_MAX;//1.0f;
        for(int colb=0;colb<N;colb++)
            delta[row*N+colb]=(DATA)rand() / (DATA)RAND_MAX;      
    }
    for(int cola=0;cola<M;cola++)
        for(int colb=0;colb<N;colb++)
            w[cola*N+colb]=(DATA)rand() / (DATA)RAND_MAX;//1.0f;
/*  -------------------------------------   */
    cudaMemcpy(d_h2h,h2h,P*M*sizeof(DATA),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,w,N*M*sizeof(DATA),cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta,delta,P*N*sizeof(DATA),cudaMemcpyHostToDevice);
    cudaMemcpy(new_delta,d_dest_delta,P*M*sizeof(DATA),cudaMemcpyHostToDevice);
    cudaMemcpy(d_thread_delta,c_host,N*M*sizeof(DATA),cudaMemcpyHostToDevice);

    backward(h2h, delta, dest_c, d_h2h, d_w, d_delta, d_thread_delta, d_dest_delta, M, N);
    
    for(int row=0;row<P;row++){
        for(int cola=0;cola<M;cola++){
            DATA temp= 0.0f;
            for(int colb=0;colb<N;colb++)
                temp+= delta[row*N+colb]*w[cola*N+colb];    
            delta_host[row*M+cola] = temp*h2h[row*M+cola]*(1-h2h[row*M+cola]);
        }
    }

    for(int row=0;row<P;row++){
        for(int colb=0;colb<N;colb++)
            for(int cola=0;cola<M;cola++)
                c_host[cola*N+colb]+= h2h[row*M+cola]* delta[row*N+colb];      
    }

    cudaMemcpy(dest_c,d_thread_delta,M*N*sizeof(DATA),cudaMemcpyDeviceToHost);
    cudaMemcpy(new_delta,d_dest_delta,P*M*sizeof(DATA),cudaMemcpyDeviceToHost);

    //printMat(c_host,M,N);
    printf("------------------------------\n");
    //printMat(dest_c,M,N);
    printf("------------------------------\n");
    matsAreEquals(dest_c,c_host,M,N);
    printf("+++\n");
    matsAreEquals(new_delta,delta_host,P,M);
    printf("+++\n");

    //printMat(new_delta,P,M);

    free(h2h);
    free(w);
    free(delta);
    free(delta_host);
    free(c_host);
    free(dest_c);
    cudaFree(d_h2h);
    cudaFree(d_w);
    cudaFree(d_delta);
    cudaFree(d_thread_delta);
    cudaFree(d_dest_delta);
    return 0;
}