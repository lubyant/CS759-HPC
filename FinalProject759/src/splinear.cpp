#include "splinear.hpp"
#include <iostream>
#include <math.h>

void matmul(double **A, const double *x, unsigned int n, double *ret);
double inner_product(const double *x1, const double *x2, unsigned int n);

void LUdecomposition(double **a, double **l, double **u, int n);

// implementation of conjugate gradient descent algorithm
void splinear_solver(double **A, const double *b, const unsigned int n,
                     double *x, int max_iter, double tol) {
  // initial residual
  double *r0 = new double[n];
  double *p0 = new double[n];
  double *w = new double[n];
  double *r = new double[n];

  // update r0 = A*x
  matmul(A, x, n, r0);

  for (unsigned int i = 0; i < n; i++) {
    r0[i] = b[i] - r0[i];
    p0[i] = r0[i];
  }

  // iterate the conjugate
  double alpha, beta;
  for (int t = 0; t < max_iter; t++) {
    matmul(A, p0, n, w);
    alpha = inner_product(r0, r0, n) / inner_product(p0, w, n);

    // assigne the update to x
    for (unsigned int i = 0; i < n; i++) {
      x[i] = x[i] + alpha * p0[i];
      r[i] = r0[i] - alpha * w[i];
    }
    if (norm(r, n) < tol)
      return;
    else {
      beta = inner_product(r, r, n) / inner_product(r0, r0, n);
      // update next iteration
      for (unsigned int i = 0; i < n; i++) {
        p0[i] = r[i] + beta * p0[i];
        r0[i] = r[i];
      }
    }
  }

  // deallocate
  delete[] r0;
  delete[] p0;
  delete[] w;
  delete[] r;
}

// ret = A*x
void matmul(double **A, const double *x, unsigned int n, double *ret) {
  double sum;
  for (unsigned int i = 0; i < n; i++) {
    sum = 0;
    for (unsigned int j = 0; j < n; j++) {
      sum += A[i][j] * x[j];
    }
    ret[i] = sum;
  }
}

// inner product
double inner_product(const double *x1, const double *x2, unsigned int n) {
  double sum = 0;
  for (unsigned int i = 0; i < n; i++)
    sum += x1[i] * x2[i];
  return sum;
}

// 2rd norm
double norm(const double *x, const unsigned int n) {
  double sum = 0;
  for (unsigned int i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  return sqrt(sum);
}

// inf norm
double norm_inf(double **A, const unsigned int n) {
  double norm = -1000000.0;
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      norm = abs(A[i][j]) > norm ? abs(A[i][j]) : norm;
    }
  }
  return norm;
}

// LU decomposition
void LUdecomposition(double **a, double **l, double **u, int n) {
  int i = 0, j = 0, k = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (j < i)
        l[j][i] = 0;
      else {
        l[j][i] = a[j][i];
        for (k = 0; k < i; k++) {
          l[j][i] = l[j][i] - l[j][k] * u[k][i];
        }
      }
    }
    for (j = 0; j < n; j++) {
      if (j < i)
        u[i][j] = 0;
      else if (j == i)
        u[i][j] = 1;
      else {
        u[i][j] = a[i][j] / l[i][i];
        for (k = 0; k < i; k++) {
          u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
        }
      }
    }
  }
}

// Sparse linear solver based on LU decomposition
void splinear_solver2(double **A, const double *b, const unsigned int n,
                      double *x) {
  double **L = new double *[n];
  double **U = new double *[n];
  double *z = new double[n];
  for (unsigned int i = 0; i < n; i++) {
    L[i] = new double[n];
    U[i] = new double[n];
  }

  // LU decomposition
  LUdecomposition(A, L, U, n);
  double sum;
  for (unsigned i = 0; i < n; i++) { // forward subtitution method
    sum = 0;
    for (unsigned p = 0; p <= i; p++)
      sum += L[i][p] * z[p];
    z[i] = (b[i] - sum) / L[i][i];
  }
  //********** FINDING X; UX=Z***********//
  for (int i = (int)n - 1; i >= 0; i--) {
    sum = 0;
    for (int k = (int)n - 1; k >= i; k--)
      sum += U[i][k] * x[k];
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

//   splinear_solver2(A, b, n, x);
//   for (unsigned int i = 0; i < n; i++) {
//     std::cout << x[i] << "\n";
//   }
//   delete[] x;
//   for (unsigned int i = 0; i < n; i++)
//     delete[] A[i];
//   delete[] A;

//   return 0;
// }