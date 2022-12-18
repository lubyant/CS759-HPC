#include "matmul.h"
#include <string.h>

typedef unsigned int uint;
void mmul1(const double *A, const double *B, double *C, const unsigned int n) {
  memset(C, 0, n * n * sizeof(double));
  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      for (uint k = 0; k < n; k++) {
        C[i * n + j] += A[i * n + k] * B[j + n * k];
      }
    }
  }
}

void mmul2(const double *A, const double *B, double *C, const unsigned int n) {
  memset(C, 0, n * n * sizeof(double));
  for (uint i = 0; i < n; i++) {
    for (uint k = 0; k < n; k++) {
      for (uint j = 0; j < n; j++) {
        C[i * n + j] += A[i * n + k] * B[j + n * k];
      }
    }
  }
}
void mmul3(const double *A, const double *B, double *C, const unsigned int n) {
  memset(C, 0, n * n * sizeof(double));
  for (uint j = 0; j < n; j++) {
    for (uint k = 0; k < n; k++) {
      for (uint i = 0; i < n; i++) {
        C[i * n + j] += A[i * n + k] * B[j + n * k];
      }
    }
  }
}
void mmul4(const std::vector<double> &A, const std::vector<double> &B,
           double *C, const unsigned int n) {
  memset(C, 0, n * n * sizeof(double));
  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      for (uint k = 0; k < n; k++) {
        C[i * n + j] += A[i * n + k] * B[j + n * k];
      }
    }
  }
}