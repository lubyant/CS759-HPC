#include "cluster.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>

/**
 * @brief version 1
 * 
 */
// void cluster(const size_t n, const size_t t, const float *arr,
//              const float *centers, float *dists) {
//               float sum = 0;
// #pragma omp parallel num_threads(t) reduction(+ : sum)
//   {
//     unsigned int tid = omp_get_thread_num();

// #pragma omp for
//     for (size_t i = 0; i < n; i++) {
//       // printf("id: %d, |%f-%f|\n", tid, arr[i], centers[tid]);
//       sum += std::fabs(arr[i] - centers[tid]);
//     }
//     dists[tid] = sum;
//   }
// }

/**
 * @brief false sharing
 * 
 */
// void cluster(const size_t n, const size_t t, const float *arr,
//              const float *centers, float *dists) {
// #pragma omp parallel num_threads(t)
//   {
//     unsigned int tid = omp_get_thread_num();
// #pragma omp for
//     for (size_t i = 0; i < n; i++) {
//       dists[tid] += std::fabs(arr[i] - centers[tid]);
//     }
//   }
// }

/**
 * @brief reduction + static schedule
 * 
 * @param n length of arr
 * @param t num of thread
 * @param arr array
 * @param centers centers array
 * @param dists distance array
 */
void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
  // parallel block
#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
  // reduction + static schedule to ease false sharing
#pragma omp for reduction(+ : dists[:t]) schedule(static, 16)
    for (size_t i = 0; i < n; i++) {
      dists[tid] += std::fabs(arr[i] - centers[tid]);
    }
  }
}

// int main(int argc, char *argv[]) {
//   // size_t n = (size_t)atoi(argv[1]);
//   // size_t t = (size_t)atoi(argv[2]);
//   size_t n = 8;
//   size_t t = 2;

//   // fill out the arr and sort it
//   // float *arr = new float[n];
//   // for(size_t i=0; i<n; i++){
//   //     arr[i] = n*(float)rand()/(float)RAND_MAX;
//   // }
//   float *arr = new float[n]{0, 1, 3, 4, 6, 6, 7, 8};
//   std::sort(arr, arr + n);

//   // fill out the centers arr
//   float *centers = new float[t];
//   for (size_t i = 1; i <= t; i++)
//     centers[i - 1] = (float)(2 * i - 1) * n / 2 / t;

//   // dists arr
//   float *dists = new float[t];
//   cluster(n, t, arr, centers, dists);

//   // print out the result
//   for (size_t i = 0; i < t; i++)
//     printf("%f\n", dists[i]);

//   // deallocate
//   delete[] arr;
//   delete[] dists;
//   delete[] centers;

//   return 0;
// }