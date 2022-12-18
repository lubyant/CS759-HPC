build: task1 task2

task1: task1.cu
	nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task1

task2: task2.cu stencil.cu
	nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task2

matmul: matmul.cpp
	g++ matmul.cpp -o matmul

stencil: stencil.cpp
	g++ stencil.cpp -o stencil

clean:
	rm task2 task1
