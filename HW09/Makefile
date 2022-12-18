task1: task1.cpp cluster.cpp
	g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

task2: task2.cpp montecarlo.cpp
	g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

task3: task3.cpp
	mpicxx task3.cpp -Wall -O3 -o task3