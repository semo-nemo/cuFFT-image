/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper/helper_functions.h"
#include "helper/helper_cuda.h"

#include <ctime>
#include <time.h> 
#include <math.h>
#include <cufft.h>
#include <fstream>

using namespace std;

using namespace cv;

typedef float2 CX;

#define rows  1936 
#define columns  2592  

    

__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 256;
    std::string inputImage = "semonemo.png";
    std::string outputImage = "fft-semonemo.png";
    std::string currentPartId = "test";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-p") == 0)
        {
            currentPartId = value;
        }
    }
    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " currentPartId: " << currentPartId << " threadsPerBlock: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, currentPartId, threadsPerBlock};
}

__host__ std::tuple<uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile)
{
    cout << "Reading Image From File\n";
    Mat img = imread(inputFile, IMREAD_COLOR);
     
    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    uchar *h_r = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *h_g = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *h_b = (uchar *)malloc(sizeof(uchar) * rows * columns);
    
    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            Vec3b intensity = img.at<Vec3b>(r, c);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            h_r[r*columns+c] = red;
            h_g[r*columns+c] = green;
            h_b[r*columns+c] = blue;
        }
    }

    return {h_r, h_g, h_b};
}
 
__host__ Mat getMat(std::string inputFile)
{
    cout << "Create Matrix to write output pixels\n";
    Mat img = imread(inputFile, IMREAD_COLOR);
     
    return img;
}

__global__ void filter_fft(CX *a, int lb, int ub)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int column_i  = i % columns;
    int row_i = i % rows;
    // remove low frequency
    if (column_i < lb || row_i < lb) {
        a[i].x = 0;
        a[i].y = 0;
    }
    // remove high frequency
    if (column_i > columns - ub || row_i > rows - ub) {
        a[i].x = 0;
        a[i].y = 0;
    }
}


int main(int argc, char *argv[])
{
    int N = 5;
    std::tuple<std::string, std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    std::string currentPartId = get<2>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<3>(parsedCommandLineArgsTuple);
    try 
    {
        auto[h_r, h_g, h_b] = readImageFromFile(inputImage);

        int SIZE = rows * columns;

        CX *temp_mat = new CX[SIZE];
        for (int i = 0; i < SIZE; i++){
            temp_mat[i].x = h_r[i];
            temp_mat[i].y = 0;
        } 

        int mem_size = sizeof(CX)* SIZE;

        cufftComplex *d_signal;
        checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size)); 
        checkCudaErrors(cudaMemcpy(d_signal, temp_mat, mem_size, cudaMemcpyHostToDevice));
    
        // CUFFT plan
        cufftHandle plan;
        cufftPlan2d(&plan, rows, columns, CUFFT_C2C);

        // Transform signal and filter
        printf("Transforming signal cufftExecR2C\n"); 
        cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD); 
        
        printf("Filter some FFT components<<< >>>\n");
        filter_fft <<< N, N >> >(d_signal,2,2); 

        // Transform signal back
        printf("Transforming signal back cufftExecC2C\n");
        cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, (int)CUFFT_INVERSE);

        CX *red_host = new CX[SIZE];
        cudaMemcpy(red_host, d_signal, sizeof(CX)*SIZE, cudaMemcpyDeviceToHost); 

        // REPEAT for blue

         
        for (int i = 0; i < SIZE; i++){
            temp_mat[i].x = h_b[i];
            temp_mat[i].y = 0;
        }  
 
        checkCudaErrors(cudaMemcpy(d_signal, temp_mat, mem_size, cudaMemcpyHostToDevice));
     
        // Transform signal and filter
        printf("Transforming signal cufftExecR2C\n");
        cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD); 
        
        printf("Filter some FFT components<<< >>>\n");
        filter_fft <<< N, N >> >(d_signal, 2, 2); 

        // Transform signal back
        printf("Transforming signal back cufftExecC2C\n");
        cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, (int)CUFFT_INVERSE);

        CX *blue_host = new CX[SIZE];
        cudaMemcpy(blue_host, d_signal, sizeof(CX)*SIZE, cudaMemcpyDeviceToHost); 

        // REPEAT for green

           
        for (int i = 0; i < SIZE; i++){
            temp_mat[i].x = h_g[i];
            temp_mat[i].y = 0;
        }  
 
        checkCudaErrors(cudaMemcpy(d_signal, temp_mat, mem_size, cudaMemcpyHostToDevice));
     
        // Transform signal and filter
        printf("Transforming signal cufftExecR2C\n");
        cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD); 
        
        printf("Filter some FFT components<<< >>>\n");
        filter_fft <<< N, N >> >(d_signal,2,2); 

        // Transform signal back
        printf("Transforming signal back cufftExecC2C\n");
        cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, (int)CUFFT_INVERSE);

        CX *green_host = new CX[SIZE];
        cudaMemcpy(green_host, d_signal, sizeof(CX)*SIZE, cudaMemcpyDeviceToHost); 
 
        delete temp_mat;
        cufftDestroy((cufftHandle)plan);
        cudaFree(d_signal); 

        Mat img = getMat(inputImage); 
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        cout << "Assign RGB intensities \n";
        for(int r = 0; r < rows; ++r)
        {
            for(int c = 0; c < columns; ++c)
            {
                img.at<Vec3b>(r, c)[0] = (float)blue_host[r*columns+c].x;
                img.at<Vec3b>(r, c)[1] = (float)green_host[r*columns+c].x;
                img.at<Vec3b>(r, c)[2] = (float)red_host[r*columns+c].x;
            }
        } 
        cout << "Write image to file \n";
        imwrite(outputImage, img, compression_params);
    }
    catch (cv::Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
     
    cout << "Finished \n";

    return 0;
}