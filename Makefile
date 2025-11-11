# Makefile for fft_image_restoration

CXX = g++
CXXFLAGS = -std=c++17 -O2 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
TARGET = fft_image_restoration
SRC = main.cpp utils.cpp fft.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
