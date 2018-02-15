#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define min(a,b) (a) < (b) ? (a) : (b)
#define BOOL int
#define blockSide 16
#define blockNum 12
#define epsilon 1e-5
#define N 58
#define M 784
#define P 680
#define DATA float
/*     OCCHIO IL PROBLEMA C'è AL CAMBIO GRIGLIA . I BLOCCHI DA GRID 0 A GRID 1 E COSì VIA TENGONO IN MEMORIA DEGLI ELEMENTI         */


/* IMPORTANTE è meglio fissare la dimensione a 16*16*16 ed in caso si sforino le matrici si setta 0 */
/* a_corner,b_corner  sono in previsione di una "sliding grid" */
__device__ void matrix_array_block(DATA* block_src_A, DATA* block_B, DATA* dest,int col1,int col2, int A_right_limit, int B_right_limit){
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int block_x = blockIdx.x*blockSide;
    int block_y = blockIdx.y*blockSide;
    int idx = t_x + block_x ; 
    int idy = t_y + block_y ;
    int pattern;
    int a_corner = blockIdx.x*blockSide;
    int b_corner = blockIdx.y*blockSide;
  

    __shared__ DATA temp_shifted_mul[blockSide][blockSide*blockSide];//può contenere diversi 0 nei casi sui bordi
    __shared__ DATA block_arr_A[blockSide*blockSide];

    block_arr_A[t_x+t_y*blockSide]=0.0f;
    


    for(int curr_patterns=0;curr_patterns<P;curr_patterns+=blockSide){
        pattern = (curr_patterns  + blockSide > P) ? (P-curr_patterns): blockSide ;

        int max_b_x = ((b_corner + blockSide) < B_right_limit) ? blockSide: (B_right_limit - b_corner);
        int max_a_x = ((a_corner + blockSide) < A_right_limit) ? blockSide: (A_right_limit - a_corner);

        DATA val = ((curr_patterns+ t_y) < P && max_b_x > t_x) ? block_B[t_y*col2 + t_x + curr_patterns*col2]:0.0f;

        //printf("t[%d][%d]->val:%f,(corner %d) [pattern:%d]+ block_B[t_x*col2 + t_y]:%f\n",idy,idx,val,a_corner,pattern,block_B[t_y*col2 + t_x]);
        for(int i=0 ;i<blockSide;i++)
            temp_shifted_mul[t_y][ t_x*blockSide + i ] =  ( (curr_patterns+ t_y) < P && max_a_x > i) ? val*block_src_A[t_y*col1 + i +curr_patterns*col1]:0.0f;
        __syncthreads();
        //il problema è qua con block
        if(t_y==0){
            for(int j=t_x*blockSide,index=0; index<blockSide;j+=1, index++ ){//j<blockSide*blockSide
                //int temp_sum=0;
                for(int i=0 ;i<pattern;i++){
                    block_arr_A[j] += temp_shifted_mul[i][j];
                // printf("t[%d][%d]->block_arr[%d]:%f------block_src_A %d---val--%d\n",idy,idx,j,block_arr_A[j],val,(t_y < pattern && max_a_x > t_x), (t_y < pattern && max_b_x > t_x));
                }
                //block_arr_A[j]+=temp_sum;
            }
        }
        __syncthreads();

    }
    if(t_x + a_corner < A_right_limit && t_y + b_corner < B_right_limit){//qui i pattern non c'etrano più nulla NON METTERE ALCUN CONTROLLO
        // printf("t[%d][%d]->block_arr[%d]:%f\n",idy,idx,t_y*blockSide+ t_x,block_arr_A[t_y*blockSide+ t_x]);
        dest[t_x+t_y*col1] = block_arr_A[t_y*blockSide+ t_x];
    }

}
/*
    fissiamo griglia su B: ogni blocco calcola 
*/
__global__ void matrix_mul(DATA* A,DATA* B, DATA* DEST,int col1,int col2, int A_right_limit, int B_right_limit){
    unsigned int t_x = threadIdx.x;
    unsigned int t_y = threadIdx.y;
    unsigned int idx = t_x + blockIdx.x ; 
    unsigned int idy = t_y + blockIdx.y ;
    int b_x = blockIdx.x*blockSide;
    int b_y = blockIdx.y*blockSide;

    matrix_array_block(A+b_x, B+b_y, DEST+b_x+b_y*col1, col1, col2, A_right_limit, B_right_limit);
    __syncthreads();
}
void optimum_grid_x(dim3* grid,int max_block,int y_limit){
    
    int x = min((M+blockSide-1)/blockSide,max_block);
    int y=max_block/x;
    int prod=x*y;
    int new_prod;
    int new_x;

    for(new_x=x, new_prod=new_x*y ; new_prod != max_block && new_x > 1 && y < y_limit ;new_x--){
        y=max_block/new_x;
        new_prod=new_x*y;
    }

    if(new_prod>prod){
        x= new_x+1;
    }

    grid->x = x;
    grid->y = y;

}
void backward(DATA *host_A, DATA* host_B, DATA* host_DEST,DATA* d_a,DATA* d_b, DATA* d_c, int col1, int col2){
    dim3 grid,block;
    /* grid.x = (M+blockSide-1)/blockSide;
    grid.y= blockNum/grid.x;  */
    optimum_grid_x(&grid,blockNum,col2/blockSide);
    block.x= blockSide;
    block.y= blockSide;
    printf("grid :%d %d\n",grid.y,grid.x);
    for(int sw_x=0; sw_x < col1; sw_x += grid.x*blockSide)
        for(int sw_y=0; sw_y < col2;sw_y += grid.y*blockSide) 
            matrix_mul<<< grid,block >>>(d_a,d_b,d_c+sw_x+sw_y*col1,col1,col2,min(col1-sw_x,grid.x*blockSide),min(col2-sw_y,grid.y*blockSide));


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
    backward(a, b, dest_c, d_a , d_b, d_c, M, N);
    
    for(int row=0;row<P;row++){
        for(int colb=0;colb<N;colb++)
            for(int cola=0;cola<M;cola++)
                c_host[colb*M+cola]+= a[row*M+cola]* b[row*N+colb];      
    }
    cudaMemcpy(dest_c,d_c,N*M*sizeof(DATA),cudaMemcpyDeviceToHost);

   // printMat(c_host,N,M);
    printf("------------------------------\n");
    //printMat(dest_c,N,M);
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