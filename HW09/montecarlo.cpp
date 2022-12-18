#include "montecarlo.h"

#define dist(x, y) ((x) * (x)) + ((y) * (y))

/**
 * @brief montecarlo algo to compute the pi
 *
 * @param n length of array
 * @param x array of x coordinates
 * @param y array of y coordinates
 * @param radius radius of circle
 * @return int counts that inside of cirle
 */
int montecarlo(const size_t n, const float *x, const float *y,
               const float radius) {
  int count = 0;
  float area = radius * radius; // optimized by removing square root
#pragma omp parallel for simd reduction(+ : count) // simd clause
  for (size_t i = 0; i < n; i++) {
    dist(x[i], y[i]) < area ? count++ : 0;
  }
  return count;
}
