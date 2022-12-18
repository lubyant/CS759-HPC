#ifndef SPLINEAR_CUDA_CUH
#define SPLINEAR_CUDA_CUH


typedef long unsigned int size_t;
typedef unsigned int uint;

/**
 * @brief sparse linear solver
 *
 * @param A
 * @param b
 * @param n
 * @param x
 * @param max_iter
 * @param tol
 */

void splinear_solver_cuda(double **A, const double *b, const unsigned int n,
                          double *x, const uint block_dim);


#endif