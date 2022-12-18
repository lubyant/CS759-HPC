#ifndef SPLINEAR_OMP_HPP
#define SPLINEAR_OMP_HPP
typedef long unsigned int size_t;

/**
 * @brief sparse linear solver with LU decomposition
 * Ax = b
 * @param A matrix A for linear system
 * @param b vector b
 * @param n size of matrix A or length of vec b
 * @param x variables
 * @param max_iter threshold for iteration to stop
 * @param tol tolerance of errors
 */

void splinear_solver_omp(double **A, const double *b, const unsigned int n,
                         double *x, const size_t threads_num);

/**
 * @brief sparse linear solver with conjugate gradient descent
 * Ax = b
 * @param A matrix A for linear system
 * @param b vector b
 * @param n size of matrix A or length of vec b
 * @param x variables
 * @param max_iter threshold for iteration to stop
 * @param tol tolerance of errors
 */
void splinear_solver2_omp(double **A, const double *b, const unsigned int n,
                          double *x, int max_iter, double tol,
                          const size_t threads_num);

/**
 * @brief 2nd norm implemented by OpenMP acceleration
 *
 * @param x array
 * @param n size of array
 * @return double 2nd norm
 */
double norm_omp(const double *x, const unsigned int n,
                const size_t threads_num);

/**
 * @brief infinity norm implmented by OpenMP acceleration
 *
 * @param A matrix
 * @param n size of matrix
 * @return double inf norm
 */
double norm_inf_omp(double **A, const unsigned int n, const size_t threads_num);
#endif