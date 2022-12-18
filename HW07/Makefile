build: task1_thrust task1_cub

task1_thrust: task1_thrust.cu
	nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task1_thrust

task1_cub: task1_cub.cu
	nvcc task1_cub.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task1_cub

task2: task2.cu count.cu
	nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task2