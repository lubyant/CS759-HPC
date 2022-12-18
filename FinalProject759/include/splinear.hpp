#ifndef SPLINEAR_HPP
#define SPLINEAR_HPP
/**
 * @brief sparse linear solver using
 * conjugate gradient descent
 * for solving the linear system
 *
 * Ax = b
 *
 * @param A matrix A
 * @param b vector B
 * @param n size of matrix A / length of vector x / length of vector b
 * @param x vector x
 * @param max_iter maximum iterations to stop to compute
 * @param tol tolerance
 */
void splinear_solver(double **A, const double *b, const unsigned int n,
                     double *x, int max_iter = 100, double tol = 0.001);

/**
 * @brief linear solver using LU decomposition
 * for solving the linear system
 *
 * Ax = b
 *
 * @param A matrix A
 * @param b vector B
 * @param n size of matrix A / length of vector x / length of vector b
 * @param x vector x
 */
void splinear_solver2(double **A, const double *b, const unsigned int n,
                      double *x);

/**
 * @brief 2nd norm
 *
 * @param x array
 * @param n size of array
 * @return double 2nd norm
 */
double norm(const double *x, const unsigned int n);

/**
 * @brief infinity norm
 *
 * @param A matrix
 * @param n size of matrix
 * @return double inf norm
 */
double norm_inf(double **A, const unsigned int n);
#endif
