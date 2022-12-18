#include "reduce.h"
#include <mpi.h>
#include <stdlib.h>
#include <chrono>
#include <stdio.h>

using namespace std;
int main(int argc, char **argv) {
  int my_rank, source, dest, tag = 0;
  // creat an array
  size_t n = (size_t)atoi(argv[1]);
  size_t t = (size_t)atoi(argv[2]);
  omp_set_num_threads(t);

  float *arr = new float[n];
  for (size_t i = 0; i < n; i++) {
    arr[i] = 2 * (float)rand() / (float)RAND_MAX - 1;
  }
  float reduction, sum;

  // timer
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, std::milli> duration_sec;

  // init MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // status
  MPI_Status status;

  // mpi
  if (my_rank == 0) { // process 0
    // sychronzed all process such that start to count time
    MPI_Barrier(MPI_COMM_WORLD);

    start = chrono::high_resolution_clock::now();

    // send to process 1
    sum = reduce(arr, 0, n / 2);
    reduction = sum;

    // sychronzed all process such that stop to count time
    MPI_Barrier(MPI_COMM_WORLD);

    end = chrono::high_resolution_clock::now();

    duration_sec = chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);
    // recieve the elapsed time from process 1
    source = 1;
    MPI_Recv(&reduction, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);

    printf("%f\n%f\n", reduction+sum, duration_sec.count());

  } else if (my_rank == 1) { // process 1
    // sychronzed all process such that start to count time
    MPI_Barrier(MPI_COMM_WORLD);
    sum = reduce(arr, n / 2 + 1, n);
    // sychronzed all process such that stop to count time
    MPI_Barrier(MPI_COMM_WORLD);

    dest = 0;
    
    MPI_Send(&sum, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
  } else {
    // in case too many threads
    printf("Wrong argument");
  }
  // close MPI
  MPI_Finalize();

  // deallocate
  delete[] arr;
  
  return 0;
}