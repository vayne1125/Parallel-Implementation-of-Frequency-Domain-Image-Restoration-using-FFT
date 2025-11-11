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
```