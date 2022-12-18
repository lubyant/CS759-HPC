#include "matmul.h"
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  using namespace std;
  size_t n = (size_t)atoi(argv[1]);
  int t = (int)atoi(argv[2]);

  // create the matrices
  float *A = new float[n * n];
  float *B = new float[n * n];
  float *C = new float[n * n];
  for (size_t i = 0; i < n * n; i++) {
    A[i] = 1;
    B[i] = 1;
    C[i] = 0;
  }

  // timer
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, std::milli> duration_sec;

  // set threads
  omp_set_num_threads(t);

  // parallel block and call the mmul function and count time
  start = std::chrono::high_resolution_clock::now();
  mmul(A, B, C, n);
  end = std::chrono::high_resolution_clock::now();

  // duration
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);

  // print out the result and elapsed time
  printf("%f\n%f\n%f\n", C[0], C[n * n - 1], duration_sec.count());

  // deallocation
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}