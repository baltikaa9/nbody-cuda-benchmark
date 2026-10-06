NVCC = nvcc
TARGET = nbody_bench.o
SRC = nbody_bench.cu

# Список GPU, доступных программе. Можно переопределить:
# make run GPUS=0,1,2,3
GPUS ?= 0,1

.PHONY: all build run gpus clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -arch=native -Xcompiler -fopenmp $(SRC) -o $(TARGET)

# Запуск с выбранными GPU. Внутри программы также выводится число найденных GPU.
run: $(TARGET)
	CUDA_VISIBLE_DEVICES=$(GPUS) ./$(TARGET)

gpus:
	nvidia-smi -L

clean:
	rm -f $(TARGET)
