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
