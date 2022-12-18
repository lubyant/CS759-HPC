#include "convolution.h"
#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  using namespace std;
  size_t n = atoi(argv[1]);
  size_t m = atoi(argv[2]);
  // size_t n = 1000;
  // size_t m = 3;

  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, milli> duration_sec;

  float *image = new float[n * n];
  float *output = new float[n * n];
  float *mask = new float[m * m];

  for (size_t i = 0; i < n * n; i++) {
    image[i] = 20 * ((float)rand()) / (float)RAND_MAX - 10;
  }

  for (size_t i = 0; i < m * m; i++) {
    mask[i] = 2 * ((float)rand()) / (float)RAND_MAX - 1;
  }

  start = chrono::high_resolution_clock::now();
  convolve(image, output, n, mask, m);
  end = chrono::high_resolution_clock::now();
  duration_sec =
      chrono::duration_cast<chrono::duration<double, milli>>(end - start);

  cout << duration_sec.count() << "\n";
  float first = output[0], last = output[n - 1];
  cout << first << "\n";
  cout << last << "\n";

  delete[] image;
  image = nullptr;
  delete[] mask;
  mask = nullptr;
  delete[] output;
  output = nullptr;
  return 0;
}