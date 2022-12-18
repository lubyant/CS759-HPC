build: task1 task2

task1: task1.cu mmul.cu
	nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -o task1

task2: task2.cu scan.cu
	nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task2

gdb: scan.cu
	nvcc -g -G scan.cu -o scan


clean:
	rm task2 task1

memcheck: scan.cu task2.cu
	nvcc scan.cu task2.cu -Xcompiler -rdynamic -lineinfo -O3 -Xcompiler -Wall -Xptxas -O3 -o task2

# memcheck: mmul.cu task1.cu
# 	nvcc mmul.cu task1.cu -Xcompiler -rdynamic -lineinfo -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -o task1

# memcheck: scan.cu
# 	nvcc scan.cu -Xcompiler -rdynamic -lineinfo -O3 -Xcompiler -Wall -Xptxas -O3 -o scan 
