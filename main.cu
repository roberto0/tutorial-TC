#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <cstdio>
#define TCSIZE 16
#define PRINTLIMIT 10

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void matmuls_tc(half* A, half* B, float *C, int n){
    int off = blockIdx.x*TCSIZE*TCSIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
   
    wmma::load_matrix_sync(a_frag, A + off, TCSIZE);
    wmma::load_matrix_sync(b_frag, B + off, TCSIZE);
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(C + off, c_frag, TCSIZE, wmma::mem_row_major);
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
       out[idx] = in[idx];
    }
 }

 __global__ void matmuls_basic(float* A, float* B, float *C, int n){
    int off = blockIdx.x * (TCSIZE*TCSIZE);
    int tid = off + (threadIdx.y*TCSIZE + threadIdx.x);
    float sum = 0.0f;
    for(int i=0; i<TCSIZE; ++i){
        sum += A[off + threadIdx.y*TCSIZE + i]*B[off + i*TCSIZE + threadIdx.x];
    }
    C[tid] = sum;
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
                m[off + i*TCSIZE + j] = (val)*rand();//(val*(k+1));
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

    //printmats(B, nmats, "[after] mat A:");

    cudaMemcpy(Ad, A, sizeof(float)*totaln, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, sizeof(float)*totaln, cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, C, sizeof(float)*totaln, cudaMemcpyHostToDevice);

    cudaSetDevice(0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block, grid;

    convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Adh, Ad, totaln);
    convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Bdh, Bd, totaln);

    if(alg == 0){
        block = dim3(TCSIZE, TCSIZE, 1);
        grid = dim3((totaln+TCSIZE*TCSIZE-1)/(TCSIZE*TCSIZE), 1, 1);
        cudaEventRecord(start);
    //    for(int i=0;i<100;++i){
            matmuls_basic<<<grid, block>>>(Ad, Bd, Cd, totaln);
            cudaDeviceSynchronize();   
    //    }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }
    if(alg == 1){
        block = dim3(TCSIZE, TCSIZE, 1);
        grid = dim3((totaln+TCSIZE*TCSIZE-1)/(TCSIZE*TCSIZE), 1, 1);
        cudaEventRecord(start);
    //    for(int i = 0; i<100; ++i){
            matmuls_basic_half<<<grid, block>>>(Adh, Bdh, Cd, totaln);
            cudaDeviceSynchronize();
    //    }
        cudaEventRecord(stop);
    }
    if(alg == 2){    
        block = dim3(TCSIZE*2,1, 1);
        grid = dim3(nmats, 1, 1);
        cudaEventRecord(start);
    //    for(int i=0;i<100; ++i){
            matmuls_tc<<<grid, block>>>(Adh, Bdh, Cd, totaln);
   //     }
        cudaEventRecord(stop);
    }

    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    //printf("%s: %f secs\n","matmuls" , time/100000.0f);
    //printf("%i, %f, %i \n",alg , time/100000.0f, nmats);
    printf("%f \n", time/100000.0f);

    cudaMemcpy(A, Ad, sizeof(float)*totaln, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, Bd, sizeof(float)*totaln, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, Cd, sizeof(float)*totaln, cudaMemcpyDeviceToHost);

    //printmats(C, nmats, "[after] mat C:");
/*    if(nmats < PRINTLIMIT){
        printmats(C, nmats, "[after] mat C:");
    }
*/
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

