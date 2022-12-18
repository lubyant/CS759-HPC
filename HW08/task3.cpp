#include "msort.h"
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  using namespace std;
  size_t n = (size_t)atoi(argv[1]);
  int t = (int)atoi(argv[2]);
  size_t ts = (size_t)atoi(argv[3]);

  // create an array and set its value between (-1000, 1000)
  int *arr = new int[n];
  for (size_t i = 0; i < n; i++) {
    arr[i] = 2000 * (float)rand() / (float)RAND_MAX - 1000;
  }

  // timer
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, std::milli> duration_sec;

  // set the number of thread
  omp_set_num_threads(t);

  // parallel block and call the function
  start = std::chrono::high_resolution_clock::now();
  msort(arr, n, ts);
  end = std::chrono::high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);

  // print out the result
  printf("%d\n%d\n%f\n", arr[0], arr[n - 1], duration_sec.count());

  // deallocate
  delete[] arr;

  return 0;
}