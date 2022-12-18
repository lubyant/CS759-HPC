build: task1 task2 task3

task1: task1.cu
	nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task1

task2: task2.cu
	nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task2

task3: task3.cu vscale.cu
	nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task3

run: build
	./task1
	./task2
	./task3 100