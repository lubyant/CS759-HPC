#include "montecarlo.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  using namespace std;

  const size_t n = (size_t)atoi(argv[1]);
  const size_t t = (size_t)atoi(argv[2]);
  const float r = 1.0;

  // generate coordinate arrays randomly
  float *x = new float[n];
  float *y = new float[n];
  for (size_t i = 0; i < n; i++) {
    x[i] = (float)rand() / (float)RAND_MAX;
    y[i] = (float)rand() / (float)RAND_MAX;
  }
  omp_set_num_threads(t);
  // timer
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, std::milli> duration_sec;
  start = std::chrono::high_resolution_clock::now();

  // run simulation 1000 time for duration
  int counts = 0;
  for (int i = 0; i < 1000; i++)
    counts += montecarlo(n, x, y, r);
  end = std::chrono::high_resolution_clock::now();

  // duration
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);

  // print out the results
  printf("%f\n%f\n", (float)4 * counts / n / 1000, duration_sec.count() / 1000);

  // deallocate
  delete[] x;
  delete[] y;
  
  return 0;
}