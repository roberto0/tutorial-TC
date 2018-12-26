#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cstdio>
#define TBLOCK 512

void initmat(int *m, int nmats, const int val){
    for(int j=0; j<nmats; ++j){
        m[j] = val;
    }
}

void printmat(int *m, int nmats){
    for(int j=0; j<nmats; ++j){
        printf("%i ", m[j]);
    }
}

__global__ void mykernel() {
    //int idx = threadIdx.x;
    /*for (int i = N / 2; i > 0; i /= 2){
        __syncthreads();
        if(idx<i){
            in[idx] += in[idx + i];
        }
        __syncthreads();
    }
    //printf("in[%i]: %i \n",idx, in[idx]);
   */ //out[0] = in[0];
    __syncthreads();
    printf("t: %i \n ",3);
    __syncthreads();
//    if(idx == 0) printf("in[%i]: %i \n",idx, in[idx]);
}

/*#define FULL_MASK 0xffffffff
__inline__ __device__
int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__global__ void deviceReduceWarpAtomicKernel(int *in, int* out, int N) {
  int sum = int(0);
  for(int i = blockIdx.x * blockDim.x + threadIdx.x;
      i < N;
      i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = warpReduceSum(sum);
  if ((threadIdx.x & (warpSize - 1)) == 0)
    atomicAdd(out, sum);
}
*/
int main(int argc, char **argv){  
    int N = 512;
    
    int *in, *out;
    int *ind, *outd;
    
    in = (int*)malloc(sizeof(int)*N);
    out = (int*)malloc(sizeof(int)*N);
    
    initmat(in, N, 1);
    initmat(out, N, 0);
    
    cudaMalloc(&ind, sizeof(int)*N);
    cudaMalloc(&outd, sizeof(int)*N);
    
    cudaMemcpy(ind, in, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(outd, out, sizeof(int)*N, cudaMemcpyHostToDevice);
    
    //printmat(in, N);
    cudaSetDevice(0);
    dim3 block, grid;
    block = dim3(TBLOCK, 1, 1);
    grid = dim3((N+TBLOCK*TBLOCK-1)/(TBLOCK*TBLOCK), 1, 1);


    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //cudaEventRecord(start);
    printf("here 1 \n ");
    mykernel<<<grid,block>>>();
    cudaDeviceSynchronize(); 
    printf("here 2 \n ");
    //cudaDeviceSynchronize();
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float time = 0;
    //cudaEventElapsedTime(&time, start, stop);
    //printf("%f \n", time/100000.0f);
    //deviceReduceKernel<<<blocks, threads>>>(in, out, N);
    //deviceReduceKernel<<<1, 1024>>>(out, out, blocks);

    free(in);
    free(out);
    cudaFree(ind);
    cudaFree(outd);

    exit(EXIT_SUCCESS);
}
