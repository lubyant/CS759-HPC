#include "scan.h"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  int n = std::atoi(argv[1]);
  float *arr = new float[n];    // input
  float *output = new float[n]; // output
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::chrono::duration<double, std::milli> duration_sec;
  for (int i = 0; i < n; i++) {
    arr[i] = 2 * ((float)rand()) / (float)RAND_MAX - 1;
  }
  start = std::chrono::high_resolution_clock::now();
  scan(arr, output, n);
  end = std::chrono::high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);
  float first = output[0], last = output[n - 1];

  std::cout << duration_sec.count() << "\n";
  std::cout << first << "\n";
  std::cout << last << "\n";

  // deallocate
  delete[] arr;
  arr = nullptr;
  delete[] output;
  output = nullptr;
  return 0;
}