# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) convertRGB.cu --std c++17 `pkg-config opencv --cflags --libs` -o convertRGB.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcufft

run:
	./convertRGB.exe $(ARGS) > output.txt

clean:
	rm -f convertRGB.exe output*.txt 