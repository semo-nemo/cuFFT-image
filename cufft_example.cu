#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper/helper_functions.h"
#include "helper/helper_cuda.h"

#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cufft.h>
#include <fstream>

using namespace std;
typedef float2 Complex;
#define rows 20 //1936/100;
#define columns 25 // 2592/100;

//Found at http://techqa.info/programming/question/36889333/cuda-cufft-2d-example


__global__ void filter_fft(Complex *a)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int column_i  = i % columns;
    int row_i = i % rows;
    // remove low frequency
    if (column_i < 5 || row_i < 5) {
        a[i].x = 0;
        a[i].y = 0;
    }
    // remove high frequency
    if (column_i > columns - 5 || row_i > rows - 5) {
        a[i].x = 0;
        a[i].y = 0;
    }
}


int main()
{ 
    int N = 5;

    int SIZE = rows * columns;


    Complex *fg = new Complex[SIZE];
    for (int i = 0; i < SIZE; i++){
        fg[i].x = 1;
        fg[i].y = 0;
    } 

    int mem_size = sizeof(Complex)* SIZE;

    cufftComplex *d_signal;
    checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size)); 
    checkCudaErrors(cudaMemcpy(d_signal, fg, mem_size, cudaMemcpyHostToDevice));
 
    // CUFFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, rows, columns, CUFFT_C2C);

    // Transform signal and filter
    printf("Transforming signal cufftExecR2C\n");
    int direction = CUFFT_FORWARD;
    cufftResult res;
    res = cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, (int)direction); 
    
    printf("Filter some FFT components<<< >>>\n");
    filter_fft <<< N, N >> >(d_signal); 

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, (int)CUFFT_INVERSE);

    Complex *result = new Complex[SIZE];
    cudaMemcpy(result, d_signal, sizeof(Complex)*SIZE, cudaMemcpyDeviceToHost);

 

    delete result, fg;
    cufftDestroy((cufftHandle)plan);
    cudaFree(d_signal); 

}