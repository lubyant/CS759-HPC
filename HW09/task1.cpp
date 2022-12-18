#include "cluster.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

std::pair<float, int> find_max(float *dists, size_t t);

int main(int argc, char *argv[]) {
  using namespace std;
  size_t n = (size_t)atoi(argv[1]);
  size_t t = (size_t)atoi(argv[2]);

  // fill out the arr and sort it
  float *arr = new float[n];
  for (size_t i = 0; i < n; i++) {
    arr[i] = n * (float)rand() / (float)RAND_MAX; // 0..n
  }

  // sort the array
  std::sort(arr, arr + n);

  // fill out the centers arr
  float *centers = new float[t];
  for (size_t i = 1; i <= t; i++)
    centers[i - 1] = (2 * i - 1) * n / 2 / t;

  // timer
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, std::milli> duration_sec;

  // dists arr
  float *dists = new float[t];
  for (size_t i = 0; i < t; i++)
    dists[i] = 0;

  // call function 10 times
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++)
    cluster(n, t, arr, centers, dists);
  end = std::chrono::high_resolution_clock::now();

  // duration
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);

  // print out the result
  std::pair<float, int> re = find_max(dists, t);
  printf("%f\n%d\n%f\n", re.first, re.second, duration_sec.count() / 10);

  // deallocate
  delete[] arr;
  delete[] dists;
  delete[] centers;

  return 0;
}

/**
 * @brief function for finding the max in dists array
 * 
 * @param dists distance array
 * @param t length of dists
 * @return std::pair<float, int> {largest_element, position of largest element} 
 */
std::pair<float, int> find_max(float *dists, size_t t) {
  float max_dist = FLT_MIN;
  int max_arg = -1;
  for (size_t i = 0; i < t; i++) {
    if (dists[i] > max_dist) {
      max_dist = dists[i];
      max_arg = i;
    }
  }
  return std::pair<float, int>({max_dist, max_arg});
}