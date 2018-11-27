#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cstdio>
#define TCSIZE 16
#define PRINTLIMIT 10

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
       out[idx] = in[idx];
    }
 }

 __global__ void matmuls_basic(float* A, float* B, float *C, int n){
    int off = blockIdx.x * (TCSIZE*TCSIZE);
    int tid = off + (threadIdx.y*TCSIZE + threadIdx.x);
    for(int i=0; i<TCSIZE; ++i){
        C[tid] += A[off + threadIdx.y*TCSIZE + i]*B[off + i*TCSIZE + threadIdx.x];
    }
}

__global__ void matmuls_basic_half(half* A, half* B, float* C, int n){
    int off = blockIdx.x * (TCSIZE*TCSIZE);
    int tid = off + (threadIdx.y*TCSIZE + threadIdx.x);
    half sum = 0.0f;
    for(int i=0; i<TCSIZE; ++i){
        sum += __hmul(A[off + threadIdx.y*TCSIZE + i],B[off + i*TCSIZE + threadIdx.x]);
    }
    C[tid] = (float)sum;
}

void initmat(float *m, int nmats, const int val){
    for(int k=0; k<nmats; ++k){
        int off = k*TCSIZE*TCSIZE;
        for(int i=0; i<TCSIZE; ++i){
            for(int j=0; j<TCSIZE; ++j){
                m[off + i*TCSIZE + j] = (val*(k+1));
            }
        }
    }
}

void printmats(float *m, int nmats, const char *msg){
    printf("%s:\n", msg);
    for(int k=0; k<nmats; ++k){
        printf("k=%i\n", k);
        int off = k*TCSIZE*TCSIZE;
        for(int i=0; i<TCSIZE; ++i){
            for(int j=0; j<TCSIZE; ++j){
                printf("%.2f ", m[off + i*TCSIZE + j]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char **argv){
    if(argc != 3){
        fprintf(stderr, "run as ./prog nmat alg\n");
        exit(EXIT_FAILURE);
    }
    int alg = atoi(argv[1]);
    int nmats = atoi(argv[2]);
    int totaln = nmats*(TCSIZE)*(TCSIZE);

    float *A,  *B,  *C;
    float *Ad, *Bd, *Cd;
    half *Adh, *Bdh, *Cdh;

    A = (float*)malloc(sizeof(float)*totaln);
    B = (float*)malloc(sizeof(float)*totaln);
    C = (float*)malloc(sizeof(float)*totaln);

    cudaMalloc(&Ad, sizeof(float)*totaln);
    cudaMalloc(&Bd, sizeof(float)*totaln);
    cudaMalloc(&Cd, sizeof(float)*totaln);
    cudaMalloc(&Adh, sizeof(half)*totaln);
    cudaMalloc(&Bdh, sizeof(half)*totaln);
    cudaMalloc(&Cdh, sizeof(half)*totaln);

    initmat(A, nmats, 1);
    initmat(B, nmats, 1);
    initmat(C, nmats, 0);

    cudaMemcpy(Ad, A, sizeof(float)*totaln, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, sizeof(float)*totaln, cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, C, sizeof(float)*totaln, cudaMemcpyHostToDevice);

    cudaSetDevice(0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block, grid;

    block = dim3(TCSIZE, TCSIZE, 1);    
    grid = dim3((totaln+TCSIZE*TCSIZE-1)/(TCSIZE*TCSIZE), 1, 1);
   
    if(alg == 0){
        cudaEventRecord(start);
        matmuls_basic<<<grid, block>>>(Ad, Bd, Cd, totaln);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        printf("%s: %f secs\n", "matmuls_basic_simple", time/1000.0f);
    }
    if(alg == 1){
        convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Adh, Ad, totaln);
        convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Bdh, Bd, totaln);
        cudaEventRecord(start);
        matmuls_basic_half<<<grid, block>>>(Adh, Bdh, Cd, totaln);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        printf("%s: %f secs\n", "matmuls_basic_simple", time/1000.0f);
    }

    cudaMemcpy(A, Ad, sizeof(float)*totaln, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, Bd, sizeof(float)*totaln, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, Cd, sizeof(float)*totaln, cudaMemcpyDeviceToHost);

    if(nmats < PRINTLIMIT){
        printmats(C, nmats, "[after] mat C:");
    }

    free(A);
    free(B);
    free(C);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    cudaFree(Adh);
    cudaFree(Bdh);

    exit(EXIT_SUCCESS);
}
