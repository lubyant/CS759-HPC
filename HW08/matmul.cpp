#include "matmul.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>

typedef unsigned int uint;

/**
 * @brief matrix multiplication
 *
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @param n size of matrix
 */
void mmul(const float *A, const float *B, float *C, const std::size_t n) {
#pragma omp parallel for // parallel at outmost loop
  for (uint i = 0; i < n; i++) {
    for (uint k = 0; k < n; k++) {
      for (uint j = 0; j < n; j++) {
        C[i * n + j] += A[i * n + k] * B[j + n * k];
      }
    }
  }
}

// int main() {
//   float *A = new float[1024];
//   float *B = new float[1024];
//   float *C = new float[1024];
//   for (int i = 0; i < 1024; i++) {
//     A[i] = 1;
//     B[i] = 1;
//     C[i] = 0;
//   }

//   omp_set_num_threads(8);
//   mmul(A, B, C, 32);

//   for (int i = 0; i < 1024; i++)
//     printf("%f\n", C[i]);
//   return 0;
// }