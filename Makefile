MODE ?= serial

CXXFLAGS = -std=c++17 -O2 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

ifeq ($(MODE), mpi)
    CXX = mpic++
    TARGET = mpi
    SRC = mpi.cpp fft/fft_mpi.cpp fft/fft_serial.cpp
else ifeq ($(MODE), simd)
    CXX = g++
    CXXFLAGS += -mavx2
    TARGET = simd
    SRC = simd.cpp fft/fft_simd.cpp fft/fft_serial.cpp
else ifeq ($(MODE), openmp)
    CXX = g++
    CXXFLAGS += -fopenmp
    LDFLAGS += -fopenmp
    TARGET = openmp
    SRC = openmp.cpp fft/fft_openmp.cpp fft/fft_serial.cpp
else ifeq ($(MODE), parallel)
    CXX = g++
    TARGET = parallel
    SRC = parallel.cpp fft/fft_parallel.cpp fft/fft_serial.cpp
else
    CXX = g++
    TARGET = serial
    SRC = serial.cpp fft/fft_serial.cpp
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f mpi simd openmp parallel serial