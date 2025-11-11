# Parallel-Implementation-of-Frequency-Domain-Image-Restoration-using-FFT
### Intro
- NYCU Parallel Programming Fall 2025
- [PP-f25 web](https://nycu-sslab.github.io/PP-f25/)

### How to use
```sh
clone https://github.com/vayne1125/Parallel-Implementation-of-Frequency-Domain-Image-Restoration-using-FFT.git
cd Parallel-Implementation-of-Frequency-Domain-Image-Restoration-using-FFT
make
# ./fft_image_restoration <img-path> <psf-length> <psf-angle>
./fft_image_restoration "./input/cat_blurred.png" 50 30 
./fft_image_restoration "./input/car_blurred.png" 40 45
```

### Change parallel mode
- using different namespace to change the mode
- `fft_parallel(all parallel mode)` mode is default option
```cpp
// in main.cpp line 37
// use serial
channels[i] = fft_serial::wienerDeblur_myfft(channel, psf, K);
// use simd
channels[i] = fft_simd::wienerDeblur_myfft(channel, psf, K);
// etc.
channels[i] = fft_xxxx::wienerDeblur_myfft(channel, psf, K);
```