#include "mmul.h"
#include <cuda_runtime.h>

/**
 * @brief matrix multiplication use cublas library
 *
 * @param handle cublasHandle
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @param n size of square matrix
 */
void mmul(cublasHandle_t handle, const float *A, const float *B, float *C,
          int n) {
  // parameter for the cublas
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  float alpha = 1, beta = 1;
  float *alpha_p = &alpha;
  float *beta_p = &beta;

  // call the cublas routine
  cublasSgemm_v2(handle, transa, transb, n, n, n, alpha_p, A, n, B, n, beta_p,
                 C, n);
}