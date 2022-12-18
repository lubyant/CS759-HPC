#include <chrono>
#include <iostream>
#include <string.h>
typedef unsigned int uint;
using namespace std;

void mmul1(const double *A, const double *B, double *C, const unsigned int n);
void mmul2(const double *A, const double *B, double *C, const unsigned int n);
void mmul3(const double *A, const double *B, double *C, const unsigned int n);

int main(int argc, char *argv[]) {
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, milli> duration_sec;

  // number of rows/cols
  uint n = (uint)atoi(argv[1]);

  // host array
  double *hA = new double[n * n];
  double *hB = new double[n * n];
  double *hC = new double[n * n]{0};

  // init array by random
  srand(time(NULL));
    for(int i=0; i<n*n; i++){
        hA[i] = 10*((double)rand()) / (double)RAND_MAX;
        hB[i] = 10*((double)rand()) / (double)RAND_MAX;
    }

  start = chrono::high_resolution_clock::now();
  mmul1(hA, hB, hC, n);
  end = chrono::high_resolution_clock::now();
  duration_sec =
      chrono::duration_cast<chrono::duration<double, milli>>(end - start);
  std::cout << duration_sec.count() << "\n";
  delete[] hA;
  delete[] hB;
  delete[] hC;

  return 0;
}

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