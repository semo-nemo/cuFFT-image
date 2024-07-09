# cuFFT on 2d image

For each RGB channel of a color image, we run 2-dimensional cuFFT twice to get the image back.
We also experiment with filtering out low and high frequency modes.


Original photo  
![Original photo in colors](https://github.com/semo-nemo/cuFFT-image/blob/main/semonemo.png?raw=true) 
Original photo  

Result after cuFFT twice
![Black and White image, which is an input to the NPP Filter](https://github.com/semo-nemo/cuFFT-image/blob/main/fft-semonemo-no-filter.png?raw=true) 
Result after cuFFT twice

Result after cuFFT twice - where top 2 and bottom 2 frequency modes are filtered out.
![Output image converted using Laplace Filter](https://github.com/semo-nemo/cuFFT-image/blob/main/fft-semonemo-2-2.png?raw=true) 
Result after cuFFT twice - where top 2 and bottom 2 frequency modes are filtered out.

Result after cuFFT twice - where top 5 and bottom 5 frequency modes are filtered out.
![Output image converted using Box Filter, code given in the lab](https://github.com/semo-nemo/cuFFT-image/blob/main/fft-semonemo-2-2.png?raw=true) 
Result after cuFFT twice - where top 5 and bottom 5 frequency modes are filtered out.

## How to build and run:
```
make clean build
make run
```

## Program Log:

Parsing CLI arguments
inputImage: semonemo.png outputImage: fft-semonemo.png currentPartId: test threadsPerBlock: 256
Reading Image From File
Rows: 1936 Columns: 2592
Transforming signal cufftExecR2C
Filter some FFT components<<< >>>
Transforming signal back cufftExecC2C
Transforming signal cufftExecR2C
Filter some FFT components<<< >>>
Transforming signal back cufftExecC2C
Transforming signal cufftExecR2C
Filter some FFT components<<< >>>
Transforming signal back cufftExecC2C
Create Matrix to write output pixels
Assign RGB intensities 
Write image to file 
Finished 

## Discussion

Running FFT twice, we get the overall shape. The colors look different from the original. When we drop high and low frequency modes, the colors get even less realistic.

We ran C2C FFT, while the RGB channels only have real values. When we convert between complex and real numbers, we dropped imaginary parts. We could instead take magnitude of complex number to get real number. 

## Discussion for future improvements

The picture size is hard coded into the code. 
