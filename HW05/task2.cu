#include "matmul.cuh"
#include <stdio.h>

// some functions to generate test cases
void matmul_int(uint, uint);    // matmul1
void matmul_float(uint, uint);  // matmul2
void matmul_double(uint, uint); // matmul3

int main(int argc, char *argv[]) {
  uint N = (uint)atoi(argv[1]);
  uint block_dim = (uint)atoi(argv[2]);
  // uint N = pow(2, 14);
  // uint block_dim = 32;

  // matmul1
  matmul_int(N, block_dim);

  // matmul2
  matmul_float(N, block_dim);

  // matmul3
  matmul_double(N, block_dim);

  return 0;
}

/**
 * @brief test case for integer matmul
 * all entries set to one for easy debugging
 * @param N size of matrix
 * @param block_dim size of block
 */
void matmul_int(uint N, uint block_dim) {
  int *A = new int[N * N];
  int *B = new int[N * N];
  int *C = new int[N * N]{0};

  // set entries to one
  for (uint i = 0; i < N * N; i++) {
    A[i] = 1;
    B[i] = 1;
  }

  // call matmul1
  matmul_1(A, B, C, N, block_dim);

  // deallocate
  delete[] A;
  delete[] B;
  delete[] C;
  A = nullptr;
  B = nullptr;
  C = nullptr;
}

/**
 * @brief test case for float matmul
 * all entries set to one for easy debugging
 * @param N size of matrix
 * @param block_dim size of block
 */
void matmul_float(uint N, uint block_dim) {
  float *A = new float[N * N];
  float *B = new float[N * N];
  float *C = new float[N * N]{0};

  // set entries to one
  for (uint i = 0; i < N * N; i++) {
    A[i] = 1;
    B[i] = 1;
  }
  
  // call matmul2
  matmul_2(A, B, C, N, block_dim);

  // deallocate
  delete[] A;
  delete[] B;
  delete[] C;
  A = nullptr;
  B = nullptr;
  C = nullptr;
}

/**
 * @brief test case for double matmul
 * all entries set to one for easy debugging
 * @param N size of matrix
 * @param block_dim size of block
 */
void matmul_double(uint N, uint block_dim) {
  double *A = new double[N * N];
  double *B = new double[N * N];
  double *C = new double[N * N]{0};
  for (uint i = 0; i < N * N; i++) {
    A[i] = 1;
    B[i] = 1;
  }

  // call matmul3
  matmul_3(A, B, C, N, block_dim);

  // deallocate
  delete[] A;
  delete[] B;
  delete[] C;
  A = nullptr;
  B = nullptr;
  C = nullptr;
}
// int main(){
//     uint N = 5;
//     uint block_dim = 2;

//     // allocate
//     float *A = new float[N*N];
//     float *B = new float[N*N];
//     float *C = new float[N*N]{0};
//     srand(time(NULL));
//     for(uint i=0; i<N*N; i++){
//         // A[i] = (double)rand()/((double)RAND_MAX);
//         // B[i] = (double)rand()/((double)RAND_MAX);
//         A[i] = i;
//         B[i] = N*N-1-i;
//     }

//     matmul_2(A,B,C,N,block_dim);
//     for(uint i=0; i<N*N; i++){
//         printf("%f\n", C[i]);
//     }
// }