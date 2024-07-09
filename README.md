# cuFFT on 2d image

For each RGB channel of a color image, we run 2-dimensional cuFFT twice to get the image back.
We also experiment with filtering out low and high frequency modes.


Original photo in colors
![Original photo in colors](https://github.com/semo-nemo/cuFFT-image/blob/main/semonemo.png?raw=true) 
Original photo in colors

Black and White image, which is an input to the NPP Filter
![Black and White image, which is an input to the NPP Filter](https://github.com/semo-nemo/cuFFT-image/blob/main/fft-semonemo-no-filter.png?raw=true) 
Black and White image, which is an input to the NPP Filter

Output image converted using Laplace Filter
![Output image converted using Laplace Filter](https://github.com/semo-nemo/cuFFT-image/blob/main/fft-semonemo-2-2.png?raw=true) 
Output image converted using Laplace Filter

Output image converted using Box Filter, code given in the lab
![Output image converted using Box Filter, code given in the lab](https://github.com/semo-nemo/cuFFT-image/blob/main/fft-semonemo-2-2.png?raw=true) 
Output image converted using Box Filter, code given in the lab
