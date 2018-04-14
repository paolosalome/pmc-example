#include <stdio.h>

#include <cuda_runtime.h>
#define min(a,b) ((a)<(b)?(a):(b))
#define MY_CUDA_CHECK(call) {                                    \
    cudaError err = call;                                                    \
    if(cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define DEFAULTBLOCKSIZE 1024
#define WARPSIZE 32
/*Warp prefix Sum*/
__device__ int ssb_warp_prefix_sum(int val,int*warpReduction) {
    int lane = threadIdx.x % WARPSIZE;
    int wid = threadIdx.x/WARPSIZE;

    int temp_val,old_val;
   /* UP_SWEEP */
    for (int offset = 2; offset <=WARPSIZE ; offset *= 2){
        temp_val=val;
        temp_val=((lane+1-offset/2)>=0)?__shfl_up(temp_val, offset/2):0;
        val += ((lane+1)%offset==0)?temp_val:0;
    }
    if((lane+1)==WARPSIZE){
        *warpReduction=val;
         val=0;
    }
    /* DOWN-SWEEP */
    for (int offset = WARPSIZE; offset >1 ; offset /= 2){// x[k+2​^(d+1)​–1]=x[k+2^​d​–1]+x[k+ 2​^(d+​1)–1] 

        temp_val=val;
        old_val=val;

        __syncthreads();

        old_val=__shfl_down(old_val, offset/2);
        val=  ( ((lane+1)%(offset/2)==0) && ((lane+1)%offset !=0) )?old_val:val;

        temp_val=((lane+1-offset/2)>=0)?__shfl_up(temp_val, offset/2):0;
        val += ((lane+1)%offset==0)?temp_val:0;
        
    }

	return val;
}

__device__ int warp_reduction(int val){
    int lane=threadIdx.x%WARPSIZE;
    int temp;
    for(int off=1; off<=WARPSIZE/2;off*=2){
        temp=__shfl_up(val,off);
        val+=(lane>=off)?temp:0;
    }

    return val;
}
__global__ void ssb_prefix_sum(int* in,int* out,int N){
    int t_x= threadIdx.x;
    int wid = threadIdx.x/WARPSIZE;
    int lane=t_x%WARPSIZE;
    int j=0;
    int val,valIn,lastLane;
    int shared_dim=DEFAULTBLOCKSIZE/WARPSIZE;//(DEFAULTBLOCKSIZE +WARPSIZE-1)/WARPSIZE;
    static __shared__ int block_prefix[DEFAULTBLOCKSIZE/WARPSIZE];
    static __shared__ int first;

    if(t_x==0){
        first=0;
    }
    do{ 
        lastLane= min(DEFAULTBLOCKSIZE,N-j);
        valIn=in[t_x+j];//((t_x+j)<N)?in[t_x+j]:0;
        /*  in block prefix[i] sono presenti la somma di tutti gli elementi del warp i
            questo valore è ricavato dall'ultimo thread del warp nella fase di prefix Sum   */
        val = ssb_warp_prefix_sum(valIn,&block_prefix[wid]);   

        __syncthreads();
        /*  effettuando una riduzione in un warp troviamo in posizione [i] la somma di tutti gli elementi
            del warp [i] e di quelli prima        */
        if(wid==0){
            int temporal= (lane< shared_dim)?block_prefix[lane]:0;
            temporal= warp_reduction(temporal);
            if(lane< shared_dim)
                block_prefix[lane] = temporal ;
        }
        /* si scrive in memoria globale tenendo conto dell'offset cumulato e l'ultimo thread 
           del blocco con indice valido aggiorna il valore di first utile al prossimo turno */
        __syncthreads();
        if(t_x < lastLane)
            out[t_x+j]=(wid==0)?val+first:val+block_prefix[wid-1]+first;
        //__syncthreads();
        if(t_x == (lastLane - 1)){
            first+= block_prefix[wid];
        }
        __syncthreads();
        j+=blockDim.x;

    }while(j<N);

}



// This function verifies the shuffle scan result, for the simple
// prefix sum case.
bool CPUverify(int *h_data, int *h_result, int n_elements)
{
    // cpu verify
    for (int i=0; i<n_elements-1; i++)
    {
        h_data[i+1] = h_data[i] + h_data[i+1];
    }

    int diff = 0;

    for (int i=0 ; i<n_elements; i++)
    {
    	//printf("%d\n",h_data[i]);
        diff += h_data[i]-h_result[i];
        //nella versione cpu aggiunge anche se stesso(prefix[x]+x)
    }
    diff-=h_data[n_elements-1];
    printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
    bool bTestResult = false;

    if (diff == 0) bTestResult = true;

    return bTestResult;
}

int main(int argc, char **argv) {
    int *h_data, *h_result;
    int *d_data,*d_out;
    int blockSize = DEFAULTBLOCKSIZE;
    int n_elements= 65536;
    int n_aligned;
    if(argc>1) {
    	n_elements = atoi(argv[1]);
    }
    n_aligned=((n_elements+blockSize-1)/blockSize)*blockSize;
    int sz = sizeof(int)*n_aligned;

    printf("Starting shfl_scan\n");

    MY_CUDA_CHECK(cudaMallocHost((void **)&h_data, sizeof(int)*n_aligned));
    MY_CUDA_CHECK(cudaMallocHost((void **)&h_result, sizeof(int)*n_elements));

    //initialize data:
    printf("Computing Simple Sum test on %d (%d) elements\n",n_elements, n_aligned);
    printf("---------------------------------------------------\n");

    printf("Initialize test data\n");
    //char line[1024];
    for (int i=0; i<n_elements; i++)
    {
        h_data[i] = i;
//        fgets(line,sizeof(line),stdin);
//        sscanf(line,"%d",&h_data[i]);
    }

    for (int i=n_elements; i<n_aligned; i++) {
	h_data[i] = 0;
    }

    printf("Scan summation for %d elements\n", n_elements);

    // initialize a timer
    cudaEvent_t start, stop;
    MY_CUDA_CHECK(cudaEventCreate(&start));
    MY_CUDA_CHECK(cudaEventCreate(&stop));
    float et = 0;
    float inc = 0;

    MY_CUDA_CHECK(cudaMalloc((void **)&d_data, sz));
    MY_CUDA_CHECK(cudaMalloc((void **)&d_out, sz));

    MY_CUDA_CHECK(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));
    //ssb_prefix_sum<<< 1,blockSize>>>(d_data,d_out,n_elements);
    ssb_prefix_sum<<< 1,blockSize>>>(d_data,d_out,n_aligned);
    MY_CUDA_CHECK(cudaEventRecord(start, 0));
    MY_CUDA_CHECK(cudaEventRecord(stop, 0));
    MY_CUDA_CHECK(cudaEventSynchronize(stop));
    MY_CUDA_CHECK(cudaEventElapsedTime(&inc, start, stop));
    et+=inc;
    MY_CUDA_CHECK(cudaMemcpy(h_result, d_out, n_elements*sizeof(int), cudaMemcpyDeviceToHost));
    printf("\n");
   /*  for(int i =0;i< n_elements;i++){
        if(i%WARPSIZE==0)
            printf("\n%d) %d\n",i/WARPSIZE,h_data[i]);
        printf("%d ",h_result[i]);
        
    }
     */
//    MY_CUDA_CHECK(cudaMemcpy(h_result, d_data, n_elements*sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nTime (ms): %f\n", et);
    printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
             n_elements, et, n_elements/(et/1000.0f)/1000000.0f);

    bool bTestResult = CPUverify(h_data, h_result, n_elements);

    MY_CUDA_CHECK(cudaFreeHost(h_data));
    MY_CUDA_CHECK(cudaFreeHost(h_result));

    return (int)bTestResult;
}