#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define min(a,b) (a) < (b) ? (a) : (b)
#define BOOL int
#define blockSide 16
#define blockNum 16
#define epsilon 1e-5
#define N 58
#define M 784
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

__inline__ __device__ DATA warpGetVal(DATA val, int offset) {
    return __shfl(val, offset);
}

/* la matrice di destinazione è col1 x col2     */
/* a_corner,b_corner  sono in previsione di una "sliding grid" */
__device__ void matrix_array_block(DATA* block_src_A, DATA* block_B, DATA* dest,int col1,int col2, int A_right_limit, int B_right_limit){
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
 /*    int block_x = blockIdx.x*blockSide;
    int block_y = blockIdx.y*blockSide;
    int idx = t_x + block_x ; 
    int idy = t_y + block_y ; */
    int pattern;
    int a_corner = blockIdx.y*blockSide;
    int b_corner = blockIdx.x*blockSide;

    __shared__ DATA temp_shifted_mul[blockSide][blockSide*blockSide];//può contenere diversi 0 nei casi sui bordi
    __shared__ DATA block_arr_A[blockSide*blockSide];
    __shared__ DATA block_A[blockSide*blockSide];
    
    block_arr_A[t_x+t_y*blockSide]=0.0f;

    for(int curr_patterns=0;curr_patterns<P;curr_patterns+=blockSide){
        pattern = (curr_patterns  + blockSide > P) ? (P-curr_patterns): blockSide ;

        int max_b_x = ((b_corner + blockSide) < B_right_limit) ? blockSide: (B_right_limit - b_corner);
        int max_a_x = ((a_corner + blockSide) < A_right_limit) ? blockSide: (A_right_limit - a_corner);

        DATA val = ((curr_patterns+ t_y) < P && max_b_x > t_x) ? block_B[t_y*col2 + t_x + curr_patterns*col2]:0.0f;
        //DATA val_ai= ((curr_patterns+ t_y) < P && max_a_x > t_x) ? block_src_A[t_y*col1 + t_x +curr_patterns*col1]:0.0f;
        //block_A[t_y*blockSide+t_x]= ((curr_patterns+ t_y) < P && max_a_x > t_x) ? block_src_A[t_y*col1 + t_x +curr_patterns*col1]:0.0f;
        
        for(int i=0 ;i<blockSide;i++){
           /*  if(t_y%2==0)//utile a spezzare il warp
                temp_shifted_mul[t_y][ t_x + i*blockSide ] =  val*warpGetVal(val_ai,i);
            else
                temp_shifted_mul[t_y][ t_x + i*blockSide ] =  val*warpGetVal(val_ai,16+i);
         */
           // temp_shifted_mul[t_y][ t_x + i*blockSide ] =  val*block_A[t_y*blockSide+i];
           temp_shifted_mul[t_y][ t_x + i*blockSide ] =  ( (curr_patterns+ t_y) < P && max_a_x > i) ? val*block_src_A[t_y*col1 + i +curr_patterns*col1]:0.0f;
        }    
        __syncthreads();
        
        if(t_y==0){
            for(int j=t_x,index=0; index<blockSide;j+=blockSide, index++ ){
                for(int i=0 ;i<pattern;i++)
                    block_arr_A[j] += temp_shifted_mul[i][j];
            }
        }
        __syncthreads();

    }
    if(t_y + a_corner < A_right_limit && t_x + b_corner < B_right_limit)
        dest[t_x+t_y*col2] = block_arr_A[t_y*blockSide+ t_x];
    
}

__global__ void matrix_mul(DATA* A,DATA* B, DATA* DEST,int col1,int col2, int A_right_limit, int B_right_limit){
    int b_x = blockIdx.x*blockSide;
    int b_y = blockIdx.y*blockSide;

    matrix_array_block(A+b_y, B+b_x, DEST+b_x+b_y*col2, col1, col2, A_right_limit, B_right_limit);
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
void backward(DATA *host_A, DATA* host_B, DATA* host_DEST,DATA* d_a,DATA* d_b, DATA* d_c, int col1, int col2){
    dim3 grid,block;
    optimum_grid_x(&grid,blockNum,col1/blockSide);
    block.x= blockSide;
    block.y= blockSide;
    printf("grid :%d %d\n",grid.y,grid.x);
    cudaEvent_t start,stop;
    startTimer(&start,&stop);
    for(int sw_x=0; sw_x < col2; sw_x += grid.x*blockSide)
        for(int sw_y=0; sw_y < col1;sw_y += grid.y*blockSide) {
            matrix_mul<<< grid,block >>>(d_a+sw_y,d_b+sw_x,d_c+sw_x+sw_y*col2,col1,col2,min(col1-sw_y,grid.y*blockSide),min(col2-sw_x,grid.x*blockSide));
            //printf("grid :%d %d----limits:%d %d\n",sw_y,sw_x,min(col1-sw_y,grid.y*blockSide),min(col2-sw_x,grid.x*blockSide));
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
    DATA *a, *b, *c_host,*dest_c;
    DATA *d_a, *d_b, *d_c;

    a=(DATA *)malloc(P*M*sizeof(DATA));
    b=(DATA *)malloc(P*N*sizeof(DATA));
    c_host=(DATA *)calloc(N*M,sizeof(DATA));
    dest_c=(DATA *)calloc(N*M,sizeof(DATA));

    cudaMalloc((void**)&d_a,P*M*sizeof(DATA));
    cudaMalloc((void**)&d_b,P*N*sizeof(DATA));
    cudaMalloc((void**)&d_c,N*M*sizeof(DATA));

    for(int row=0;row<P;row++){
        for(int cola=0;cola<M;cola++)
            a[row*M+cola]=(DATA)rand() / (DATA)RAND_MAX;//1.0f;
        for(int colb=0;colb<N;colb++)
            b[row*N+colb]=(DATA)rand() / (DATA)RAND_MAX;      
    }
    cudaMemcpy(d_a,a,P*M*sizeof(DATA),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,P*N*sizeof(DATA),cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,c_host,N*M*sizeof(DATA),cudaMemcpyHostToDevice);
    backward(a, b, dest_c, d_a , d_b, d_c, M, N);
    
    for(int row=0;row<P;row++){
        for(int colb=0;colb<N;colb++)
            for(int cola=0;cola<M;cola++)
                c_host[cola*N+colb]+= a[row*M+cola]* b[row*N+colb];      
    }
    cudaMemcpy(dest_c,d_c,N*M*sizeof(DATA),cudaMemcpyDeviceToHost);

    //printMat(c_host,M,N);
    printf("------------------------------\n");
    //printMat(dest_c,M,N);
    printf("------------------------------\n");
    matsAreEquals(dest_c,c_host,M,N);
    

    free(a);
    free(b);
    free(c_host);
    free(dest_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}