#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define BOOL int
#define blockSide 16
#define blockNum 12
#define epsilon 1e-5
#define N 32
#define M 64
#define P 256
#define DATA float
/* IMPORTANTE è meglio fissare la dimensione a 16*16*16 ed in caso si sforino le matrici si setta 0 */
__device__ void matrix_array_block(DATA* block_src_A,DATA* block_B,DATA* dest,int pattern,int col1,int col2){
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int idx = t_x + blockIdx.x*blockSide ; 
    int idy = t_y + blockIdx.y*blockSide ;

    __shared__ DATA temp_shifted_mul[blockSide][blockSide*blockSide];//può contenere diversi 0 nei casi sui bordi
    __shared__ DATA block_arr_A[blockSide*blockSide];
    DATA val = block_B[t_x*col2 + t_y];
    for(int i=0 ;i<blockSide;i++)
        temp_shifted_mul[t_y][ t_x + i*blockSide  ] = val*block_src_A[t_y*col1+i];
        //con i warp si può rendere più veloce l'accesso ad A in quanto si limita l'accesso alla memoria globale
        //temp_shifted_mul[t_y*col1*col2 + t_x + i*col1  ] = block_src_A[t_y*col1+i]*val;
    __syncthreads();
    //fatta la "rigona" bisogna ridurre le colonne
    if(0==t_y){
        for(int j=t_x; j<blockSide*blockSide;j+=blockSide ){
            for(int i=0 ;i<pattern;i++)
                block_arr_A[j] += temp_shifted_mul[i][j];
            printf("t[%d][%d]->block_arr[%d]:%f\n",idy,idx,j,block_arr_A[j]);
        }
    }
    __syncthreads();

    
    dest[idx+idy*col1] = block_arr_A[t_y*blockSide+ t_x];
}
/*
    fissiamo griglia su B: ogni blocco calcola 
*/
__global__ void matrix_mul(DATA* A,DATA* B, DATA* DEST){
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    /* int idx = t_x + blockIdx.x*blockSide ; 
    int idy = t_y + blockIdx.y*blockSide ;
     */
    int a_corner = blockIdx.x*blockSide;
    int b_corner = blockIdx.y*blockSide;
   /*  __shared__ DATA m_temp[blockSide*blockSide];
    __shared__ DATA A_temp[blockSide][blockSide];
    __shared__ DATA B_temp[blockSide][blockSide]; */

    for(int a_idx=a_corner,b_idx=b_corner; a_idx<P*M; a_idx+=blockSide*M,b_idx+=blockSide*N ){
     /*    A_temp[t_y][t_x] = A[a_idx + t_x + t_y*M ];
        B_temp[t_y][t_x] = B[b_idx + t_y + t_x*N ]; */
        __syncthreads();//+ blockIdx.y*blockSide*M+ blockIdx.x*blockSide
        matrix_array_block(A+a_idx, B+b_idx, DEST, blockSide,M,N);        
    }
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
            a[row*M+cola]=1.0f;
        for(int colb=0;colb<N;colb++)
            b[row*N+colb]=1.0f;      
    }
    cudaMemcpy(d_a,a,P*M*sizeof(DATA),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,P*N*sizeof(DATA),cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,c_host,N*M*sizeof(DATA),cudaMemcpyHostToDevice);
    dim3 grid,block;
    grid.x = M/blockSide;
    grid.y= N/blockSide;
    block.x= blockSide;
    block.y= blockSide;
    //printf("grid %d,%d",grid.x,grid.y);
    matrix_mul<<<grid,block >>>(d_a,d_b,d_c);
    for(int row=0;row<P;row++){
        for(int colb=0;colb<N;colb++)
            for(int cola=0;cola<M;cola++)
                c_host[colb*M+cola]+= a[row*M+cola]* b[row*N+colb];      
    }
    cudaMemcpy(dest_c,d_c,N*M*sizeof(DATA),cudaMemcpyDeviceToHost);

    printMat(c_host,N,M);
    printf("------------------------------\n");
    printMat(dest_c,N,M);
    printf("------------------------------\n");
    matsAreEquals(dest_c,c_host,N,M);

    free(a);
    free(b);
    free(c_host);
    free(dest_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}