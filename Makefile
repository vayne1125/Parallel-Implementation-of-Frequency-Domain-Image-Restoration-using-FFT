MODE ?= serial

CXXFLAGS = -std=c++17 -O2 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

ifeq ($(MODE), mpi)
    CXX = mpic++
    TARGET = mpi
    SRC = mpi.cpp fft/fft_mpi.cpp fft/fft_serial.cpp
else ifeq ($(MODE), simd)
    CXX = g++
    CXXFLAGS += -mavx2 -mfma
    TARGET = simd
    SRC = simd.cpp fft/fft_simd.cpp fft/fft_serial.cpp
else ifeq ($(MODE), mpi_simd)
    CXX = mpic++
    CXXFLAGS += -mavx2 -mfma
    TARGET = mpi_simd
    SRC = mpi_simd.cpp fft/fft_mpi_simd.cpp fft/fft_serial.cpp
else ifeq ($(MODE), openmp)
    CXX = g++
    CXXFLAGS += -fopenmp
    LDFLAGS += -fopenmp
    TARGET = openmp
    SRC = openmp.cpp fft/fft_openmp.cpp fft/fft_serial.cpp
else ifeq ($(MODE), parallel)
    CXX = g++
    TARGET = parallel
    SRC = parallel.cpp fft/fft_parallel.cpp fft/fft_serial.
else ifeq ($(MODE), gpu)
    CXX = g++
    NVCC = nvcc

    TARGET = gpu

    SRCS_CPP = gpu.cpp fft/fft_serial.cpp
    SRCS_CU  = fft/fft_gpu.cu

    OBJS = $(SRCS_CPP:.cpp=.o) $(SRCS_CU:.cu=.o)

    OPENCV_FLAGS = `pkg-config --cflags opencv4`
    OPENCV_LIBS  = `pkg-config --libs opencv4`

    CUDA_FLAGS = -O3 -w -Xcompiler -fopenmp
	CUDA_LIBS	= -lcudart -lcufft

    CXXFLAGS = -std=c++17 -O2 -fopenmp

    all: $(TARGET)
    $(TARGET): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(OPENCV_LIBS) $(CUDA_LIBS)
    %.o: %.cpp
		$(CXX) $(CXXFLAGS) $(OPENCV_FLAGS) -c $< -o $@
    %.o: %.cu
		$(NVCC) $(CUDA_FLAGS) $(OPENCV_FLAGS) -c $< -o $@
else
    CXX = g++
    TARGET = serial
    SRC = serial.cpp fft/fft_serial.cpp
endif

ifneq ($(MODE), gpu)
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)
endif

clean:
	rm -f mpi simd mpi_simd openmp parallel serial *.o fft/*.o
