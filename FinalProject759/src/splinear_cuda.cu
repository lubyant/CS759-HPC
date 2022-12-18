#include "splinear_cuda.cuh"
#include <cuda.h>
#include <stdio.h>

void LUdecomposition_cuda(double **a, double **l, double **u, uint n,
                          const uint block_dim);
__host__ void LUdecomposition_kernel(double *a, double *l, double *u, uint n,
                                     const uint block_dim);
__global__ void reduction_kernel(double *a, double *l, double *u, uint n,
                                 uint i);
void splinear_solver_cuda(double **A, const double *b, const unsigned int n,
                          double *x, const uint block_dim) {
  double **L = new double *[n];
  double **U = new double *[n];
  double *z = new double[n];
  for (unsigned int i = 0; i < n; i++) {
    L[i] = new double[n];
    U[i] = new double[n];
  }

  // LU decomposition
  LUdecomposition_cuda(A, L, U, n, block_dim);

  double sum;
  for (unsigned i = 0; i < n; i++) { // forward subtitution method
    sum = 0;
    for (unsigned p = 0; p <= i; p++) {
      sum += L[i][p] * z[p];
    }

    z[i] = (b[i] - sum) / L[i][i];
  }
  //********** FINDING X; UX=Z***********//

  for (int i = (int)n - 1; i >= 0; i--) {
    sum = 0;
    for (int k = (int)n - 1; k >= i; k--) {
      sum += U[i][k] * x[k];
    }

    x[i] = (z[i] - sum) / U[i][i];
  }

  // deallocate
  for (unsigned int i = 0; i < n; i++) {
    delete[] L[i];
    delete[] U[i];
  }
  delete[] L;
  delete[] U;
  delete[] z;
}

void LUdecomposition_cuda(double **a, double **l, double **u, uint n,
                          const uint block_dim) {
  // pointer of host
  double *h_a = new double[n * n];
  double *h_l = new double[n * n];
  double *h_u = new double[n * n];

  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      h_a[i * n + j] = a[i][j];
      h_l[i * n + j] = l[i][j];
      h_u[i * n + j] = u[i][j];
    }
  }

  // pointer of device
  double *d_a, *d_l, *d_u;

  // allocate device memory
  cudaMalloc((void **)&d_a, n * n * sizeof(double));
  cudaMalloc((void **)&d_l, n * n * sizeof(double));
  cudaMalloc((void **)&d_u, n * n * sizeof(double));

  // copy from host to device
  cudaMemcpy(d_a, h_a, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l, h_l, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, h_u, n * n * sizeof(double), cudaMemcpyHostToDevice);

  // call kernel function
  LUdecomposition_kernel(d_a, d_l, d_u, n, block_dim);

  // copy from device to host
  cudaMemcpy(h_l, d_l, n * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_u, d_u, n * n * sizeof(double), cudaMemcpyDeviceToHost);

  // return the results
  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      l[i][j] = h_l[i * n + j];
      u[i][j] = h_u[i * n + j];
    }
  }

  // deallocate
  cudaFree(d_a);
  cudaFree(d_l);
  cudaFree(d_u);
  delete[] h_a;
  delete[] h_l;
  delete[] h_u;
}

__host__ void LUdecomposition_kernel(double *a, double *l, double *u, uint n,
                                     const uint block_dim) {
  for (uint i = 0; i < n; i++) {
    reduction_kernel<<<(n + block_dim - 1) / block_dim, block_dim>>>(a, l, u, n,
                                                                     i);
  }
}

__global__ void reduction_kernel(double *a, double *l, double *u, uint n,
                                 uint i) {
  // idx
  uint tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    if (tid < i)
      l[tid * n + i] = 0;
    else {
      l[tid * n + i] = a[tid * n + i];
      for (uint k = 0; k < i; k++) {
        l[tid * n + i] -= l[tid * n + k] * u[k * n + i];
      }
    }

    // sychronized threads for U
    __syncthreads();

    if (tid < i)
      u[i * n + tid] = 0;
    else if (tid == i)
      u[i * n + tid] = 1;
    else {
      u[i * n + tid] = a[i * n + tid] / l[i * n + i];
      for (uint k = 0; k < i; k++) {
        u[i * n + tid] -= l[i * n + k] * u[k * n + tid] / l[i * n + i];
      }
    }
  }
}

// /**
//  * @brief test case
//  *
//  * @return int
//  */
// int main() {
//   const unsigned int n = 10;
//   double *x = new double[n];
//   double **A = new double *[n];
//   double *b = new
//   double[n]{9.77777778,  15.11111111, 18.22222222, 21.33333333,
//                             24.44444444, 27.55555556, 30.66666667, 33.77777778,
//                             36.88888889, 29.22222222};

//   for (unsigned int i = 0; i < 10; i++) {
//     // x[i] = 3 + 0.7 * i;
//     x[i] = 0;
//   }

//   A[0] = new double[n]{2., 1., 0., 0., 0., 0., 0., 0., 0., 0.};
//   A[1] = new double[n]{1., 2., 1., 0., 0., 0., 0., 0., 0., 0.};
//   A[2] = new double[n]{0., 1., 2., 1., 0., 0., 0., 0., 0., 0.};
//   A[3] = new double[n]{0., 0., 1., 2., 1., 0., 0., 0., 0., 0.};
//   A[4] = new double[n]{0., 0., 0., 1., 2., 1., 0., 0., 0., 0.};
//   A[5] = new double[n]{0., 0., 0., 0., 1., 2., 1., 0., 0., 0.};
//   A[6] = new double[n]{0., 0., 0., 0., 0., 1., 2., 1., 0., 0.};
//   A[7] = new double[n]{0., 0., 0., 0., 0., 0., 1., 2., 1., 0.};
//   A[8] = new double[n]{0., 0., 0., 0., 0., 0., 0., 1., 2., 1.};
//   A[9] = new double[n]{0., 0., 0., 0., 0., 0., 0., 0., 1., 2.};
//   splinear_solver_cuda(A, b, n, x);
//   for (unsigned int i = 0; i < n; i++) {
//     printf("%f\n", x[i]);
//   }
//   delete[] x;
//   for (unsigned int i = 0; i < n; i++)
//     delete[] A[i];
//   delete[] A;

//   return 0;
// }