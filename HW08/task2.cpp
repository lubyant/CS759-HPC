#include "convolution.h"
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  using namespace std;
  size_t n = (size_t)atoi(argv[1]);
  int t = (int)atoi(argv[2]);

  // create the matrices. For the sake of debugging
  // image matrix are all 1 and mask are 1 to 9
  float *image = new float[n * n];
  float *output = new float[n * n];
  float *mask = new float[3 * 3]{1, 2, 3, 4, 5, 6, 7, 8, 9}; 
  for (size_t i = 0; i < n * n; i++) {
    image[i] = 1;
    output[i] = 0;
  }

  // timer
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, std::milli> duration_sec;

  // set the number of thread
  omp_set_num_threads(t);

  // parallel block and call the function
  start = std::chrono::high_resolution_clock::now();
  // #pragma omp parallel num_threads(t)
  convolve(image, output, n, mask, 3);
  end = std::chrono::high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);
  
  // print out the result and elapsed time
  printf("%f\n%f\n%f\n", output[0], output[n * n - 1], duration_sec.count());

  // deallocate
  delete[] image;
  delete[] output;
  delete[] mask;
  
  return 0;
}