# Parallel-Implementation-of-Frequency-Domain-Image-Restoration-using-FFT
### Intro
- NYCU Parallel Programming Fall 2025
- [PP-f25 web](https://nycu-sslab.github.io/PP-f25/)

### How to use
```sh
clone https://github.com/vayne1125/Parallel-Implementation-of-Frequency-Domain-Image-Restoration-using-FFT.git
cd Parallel-Implementation-of-Frequency-Domain-Image-Restoration-using-FFT
make MODE=parallel
# ./parallel <img-path> <psf-length> <psf-angle>
./parallel "./input/cat_blurred.png" 50 30 
./parallel "./input/car_blurred.png" 40 45
```

### Change parallel mode
- serial
- simd
- openmp
- mpi
- parallel(fastest)

#### SERIAL
```sh
make MODE=serial
# ./serial <img-path> <psf-length> <psf-angle>
./serial "./input/cat_blurred.png" 50 30 
```

#### SIMD
```sh
make MODE=simd
# todo
```

#### OpenMP
```sh
make MODE=openmp
# todo
./openmp "./input/cat_blurred.png" 50 30
```

#### MPI
```sh
make MODE=mpi
# mpirun -np <num_procs> ./mpi <img-path> <psf-length> <psf-angle>
mpirun -np 4 ./mpi "./input/cat_blurred.png" 50 30
```