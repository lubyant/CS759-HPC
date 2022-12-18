build: task1 task2

task1: task1.cu reduce.cu
	nvcc task1.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task1

task2: task2.cu matmul.cu
	nvcc task2.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task2

gdb: task2.cu matmul.cu
	nvcc -g -G task2.cu matmul.cu -o task1

task22: task22.cu matmul2.cu
	nvcc task22.cu matmul2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task2

clean:
	rm task2 task1

memcheck: task2.cu matmul.cu
	nvcc task2.cu matmul.cu -Xcompiler -rdynamic -lineinfo -O3 -Xcompiler -Wall -Xptxas -O3 -o task2
