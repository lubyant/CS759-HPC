#include "reduce.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
int main(int argc, char **argv) {
  // creat an array
  size_t n = (size_t)atoi(argv[1]);
  size_t t = (size_t)atoi(argv[2]);
  omp_set_num_threads(t);

  float *arr = new float[n];
  for (size_t i = 0; i < n; i++) {
    arr[i] = 2 * (float)rand() / (float)RAND_MAX - 1;
  }
  float sum;

  // timer
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, std::milli> duration_sec;

  start = chrono::high_resolution_clock::now();

  // use omp to reduce the whole array
  sum = reduce(arr, 0, n);

  end = chrono::high_resolution_clock::now();

  duration_sec =
      chrono::duration_cast<std::chrono::duration<double, std::milli>>(end -
                                                                       start);

  printf("%f\n%f\n", sum, duration_sec.count());

  // deallocate
  delete[] arr;

  return 0;
}