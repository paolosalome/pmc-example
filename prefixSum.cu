#include <stdio.h>

#include <cuda_runtime.h>

#define MY_CUDA_CHECK(call) {                                    \
    cudaError err = call;                                                    \
    if(cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define DEFAULTBLOCKSIZE 1024

/*Warp prefix Sum*/
__inline__ __device__ REAL ssd_warp_prefix_sum(int val,int prev,int*warpReduction) {
    int lane = threadIdx.x % warpSize;
    int temp;
   //da d = 0 fino a d=4(ossia log2(WARPSIZE-1)) offset= 2^d+1[2,4,8,16,32]
    for (int offset = 2; offset <warpSize ; offset *= 2){// x[k+2​^(d+1)​–1]=x[k+2^​d​–1]+x[k+ 2​^(d+​1)–1] 
        if((lane+1)%offset==0)//k+2^(d+1) -1
            val += __shfl_down(val, offset/2);
        __synchWarp();
    }
    if((lane+1)==warpSize){
        &warpReduction=val;
        val=0;
    }
    
    for (int offset = warpSize; offset >1 ; offset /= 2){// x[k+2​^(d+1)​–1]=x[k+2^​d​–1]+x[k+ 2​^(d+​1)–1] 
        if((lane+1)%offset==0){
            temp=val;
            val += __shfl_down(val, offset/2);
        }
        __synchWarp();
        if((lane+1)==offset/2)
            val= __shfl_up(temp,offset/2);
        __synchWarp();       
    }
    
	return val+prev;
}

/*Block prefix Sum*/
__inline__ __device__ REAL blockReduceSum(int val) {

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



__global__ void ssb_prefix_sum(int *d_data, int n_elements) {
    scorrere array per effettuare la prefixSum per ogni pezzo lungo DEFAULTBLOCKSIZE
        spezzare il blocco in parti lunge warp size e quindi effettuare la prefixSum sui warp
        sincronizzare e passare d_data[warpsize_prec] al warp su cui effettuare la prefix. alla fine della quale ogni thread aggiunge tale valore al valore sulla prefix

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
//	printf("%d\n",h_result[i]);
        diff += h_data[i]-h_result[i];
    }

    printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
    bool bTestResult = false;

    if (diff == 0) bTestResult = true;

    return bTestResult;
}

int main(int argc, char **argv) {
    int *h_data, *h_result;
    int *d_data;
    int blockSize = DEFAULTBLOCKSIZE;
    int n_elements=65536;
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
    char line[1024];
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
    MY_CUDA_CHECK(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));
    ssb_prefix_sum(d_data,n_elements);
    MY_CUDA_CHECK(cudaEventRecord(start, 0));
    MY_CUDA_CHECK(cudaEventRecord(stop, 0));
    MY_CUDA_CHECK(cudaEventSynchronize(stop));
    MY_CUDA_CHECK(cudaEventElapsedTime(&inc, start, stop));
    et+=inc;

    MY_CUDA_CHECK(cudaMemcpy(h_result, d_data, n_elements*sizeof(int), cudaMemcpyDeviceToHost));
    printf("Time (ms): %f\n", et);
    printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
             n_elements, et, n_elements/(et/1000.0f)/1000000.0f);

    bool bTestResult = CPUverify(h_data, h_result, n_elements);

    MY_CUDA_CHECK(cudaFreeHost(h_data));
    MY_CUDA_CHECK(cudaFreeHost(h_result));

    return (int)bTestResult;
}