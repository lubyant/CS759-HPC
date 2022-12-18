#include "field.hpp"
#include "splinear.hpp"
#include "splinear_omp.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdio.h>

// timer
std::chrono::high_resolution_clock::time_point start;
std::chrono::high_resolution_clock::time_point end;
std::chrono::duration<double, std::milli> duration_sec;

void velocity_predict_u(Mesh *field_u, const Mesh *field_v, const double dt,
                        const size_t threads_num);

void velocity_predict_v(const Mesh *field_u, Mesh *field_v, const double dt,
                        const size_t threads_num);

void boundary_corr(Mesh *field_u, Mesh *field_v);

void create_rhs(const Mesh *field_u, const Mesh *field_v, const double dt,
                Mesh *field_r);

void pressure_poisson_matrix(const Mesh *field_u, const Mesh *field_v,
                             const double dt, Mesh *field_p,
                             const size_t threads_num);

void correct_step(Mesh *field_u, Mesh *field_v, const double dt,
                  const Mesh *field_p, const size_t threads_num);

// correction step for u velocity
void velocity_predict_u(Mesh *field_u, const Mesh *field_v, const double dt,
                        const size_t threads_num) {
  // parameter configuration
  unsigned int imin = field_u->imin;
  unsigned int imax = field_u->imax;
  unsigned int jmin = field_v->imin;
  unsigned int jmax = field_v->imax;
  unsigned int n = field_u->n;
  double dxi = field_u->dxi;
  double dyi = field_v->dxi;
  double **u = field_u->mat;
  double **v = field_v->mat;

  // create a buffer matrix
  double **u_s = new double *[n];
  for (unsigned int i = 0; i < n; i++) {
    u_s[i] = new double[n];
  }

  // copy to the buffer
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      u_s[i][j] = u[i][j];
    }
  }

  // viscosity
  double nu = field_u->nu;

  // discretization
  double v_cur = 0, lap_op = 0;

  // parallel block
#pragma omp parallel for simd num_threads(threads_num)
  for (unsigned int j = jmin; j < jmax + 1; j++) {
    for (unsigned int i = imin + 1; i < jmax + 1; i++) {
      v_cur = (v[i - 1][j] + v[i - 1][j + 1] + v[i][j] + v[i][j + 1]) / 4;

      lap_op = (u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]) * dxi * dxi +
               (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) * dyi * dyi;

      u_s[i][j] =
          u[i][j] + dt * (nu * lap_op -
                          u[i][j] * (u[i + 1][j] - u[i - 1][j]) * 0.5 * dxi -
                          v_cur * (u[i][j + 1] - u[i][j - 1]) * 0.5 * dyi);
    }
  }

  // update velocity pointer
  field_u->mat = u_s;

  // deallocate
  for (unsigned int i = 0; i < n; i++)
    delete[] u[i];
  delete[] u;
}

void boundary_corr(Mesh *field_u, Mesh *field_v) {
  // parameter configuration
  unsigned int n = field_u->n;
  unsigned int imax = field_u->imax;
  unsigned int imin = field_u->imin;
  unsigned int jmax = field_v->imax;
  unsigned int jmin = field_v->imin;

  // create boundary condition
  double **u_b = field_u->mat;
  double **v_b = field_v->mat;

  // u velocity boundary condition
  for (unsigned int i = 0; i < n; i++) {
    // bottom
    u_b[i][jmin - 1] = 0 - u_b[i][jmin];
    // top
    u_b[i][jmax + 1] = 2 * 1 - u_b[i][jmax];
    // left
    // u_b[imin,:] = 0
    // right
    // u_b[imax,:] = 0

    // u velocity boundary condition
    // bottom
    // v_b[:,jmin] = 0
    // top
    // v_b[:,jmax] = 0
    // left
    v_b[imin - 1][i] = 0 - v_b[imin][i];
    // right
    v_b[imax + 1][i] = 0 - v_b[imax][i];
  }
}
// correction step for v velocity
void velocity_predict_v(const Mesh *field_u, Mesh *field_v, const double dt,
                        const size_t threads_num) {
  // parameter configuration
  unsigned int imin = field_u->imin;
  unsigned int imax = field_u->imax;
  unsigned int jmin = field_v->imin;
  unsigned int jmax = field_v->imax;
  unsigned int n = field_u->n;
  double dxi = field_u->dxi;
  double dyi = field_v->dxi;
  double **u = field_u->mat;
  double **v = field_v->mat;

  // create a buffer matrix
  double **v_s = new double *[n];
  for (unsigned int i = 0; i < n; i++)
    v_s[i] = new double[n];

  // copy to the buffer
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      v_s[i][j] = v[i][j];
    }
  }

  // viscosity
  double nu = field_v->nu;

  // discretization
  double u_cur = 0, lap_op = 0;

  // parallel block
#pragma omp parallel for simd num_threads(threads_num)
  for (unsigned int j = jmin; j < jmax + 1; j++) {
    for (unsigned int i = imin + 1; i < jmax + 1; i++) {
      u_cur = (u[i - 1][j] + u[i - 1][j + 1] + u[i][j] + u[i][j + 1]) / 4;

      lap_op = (v[i + 1][j] - 2 * v[i][j] + v[i - 1][j]) * dxi * dxi +
               (v[i][j + 1] - 2 * v[i][j] + v[i][j - 1]) * dyi * dyi;

      v_s[i][j] =
          v[i][j] +
          dt * (nu * lap_op - u_cur * (v[i + 1][j] - v[i - 1][j]) * 0.5 * dxi -
                v[i][j] * (v[i][j + 1] - v[i][j - 1]) * 0.5 * dyi);
    }
  }

  // update v velocity field pointer
  field_v->mat = v_s;

  // deallocate
  for (unsigned int i = 0; i < n; i++)
    delete[] v[i];
  delete[] v;
}

// create right hand side of possion function
void create_rhs(const Mesh *field_u, const Mesh *field_v, const double dt,
                Mesh *field_r) {
  // parameter configuration
  unsigned int n = 0;
  unsigned int imin = field_u->imin;
  unsigned int imax = field_u->imax;
  unsigned int jmin = field_v->imin;
  unsigned int jmax = field_v->imax;
  double dxi = field_u->dxi;
  double dyi = field_v->dxi;
  double cur_r;
  double **u_s = field_u->mat;
  double **v_s = field_v->mat;
  double rho = field_r->rho;

  // flatten the matrix to squash vector
  for (unsigned int j = jmin; j < jmax + 1; j++) {
    for (unsigned int i = imin; i < imax + 1; i++) {
      cur_r = (-rho / dt) * ((u_s[i + 1][j] - u_s[i][j]) * dxi +
                             (v_s[i][j + 1] - v_s[i][j]) * dyi);
      field_r->arr[n++] = cur_r;
    }
  }
}

// create the pressure poisson matrix by matrix method
void pressure_poisson_matrix(const Mesh *field_u, const Mesh *field_v,
                             const double dt, Mesh *field_p,
                             const size_t threads_num) {
  // parameter configuration
  unsigned int n = field_p->n - 1;
  unsigned int imin = field_u->imin;
  unsigned int imax = field_u->imax;
  unsigned int jmin = field_v->imin;
  unsigned int jmax = field_v->imax;
  Mesh lhs(n * n, n * n);
  Mesh rhs(n);
  double **mat = lhs.mat;
  double dxi = field_u->dxi;
  double dyi = field_v->dxi;

  // create the poisson matrix
#pragma omp parallel for num_threads(threads_num)
  for (int j = 0; j < (int)n; j++) {
    for (int i = 0; i < (int)n; i++) {
      mat[i + j * n][i + j * n] = 2 * dxi * dxi + 2 * dyi * dyi;
      for (int ii = i - 1; ii < i + 2; ii += 2) {
        if (ii >= 0 and ii < n)
          mat[i + (j)*n][ii + (j)*n] = -dxi * dxi;
        else
          mat[i + (j)*n][i + (j)*n] -= dxi * dxi;
      }
      for (int jj = j - 1; jj < j + 2; jj += 2) {
        if (jj >= 0 and jj < n)
          mat[i + (j)*n][i + (jj)*n] = -dyi * dyi;
        else
          mat[i + (j)*n][i + (j)*n] -= dyi * dyi;
      }
    }
  }

  // change the first line of sparse matrix
  for (unsigned int i = 0; i < n * n; i++)
    mat[0][i] = 0;
  mat[0][0] = 1;

  // create a vector for right hand side of poisson function
  create_rhs(field_u, field_v, dt, &rhs);

  // linear solver
  double *ans = new double[n * n];
  splinear_solver_omp(mat, rhs.arr, n * n, ans, threads_num);

  // update the pressure field by the results of linear solver
  unsigned int count = 0;
  for (unsigned int j = jmin; j < jmax + 1; j++) {
    for (unsigned int i = imin; i < imax + 1; i++) {
      field_p->mat[i][j] = ans[count++];
    }
  }

  // deallocate
  delete[] ans;
}

// update the velocity field by pressure
void correct_step(Mesh *field_u, Mesh *field_v, const double dt,
                  const Mesh *field_p, const size_t threads_num) {
  // field dimension
  unsigned int n = field_u->n;

  // pressure matrix
  double **P = field_p->mat;

  // allocate two velocities fields for updating
  double **u_new = new double *[n];
  for (unsigned int i = 0; i < n; i++)
    u_new[i] = new double[n];
  double **v_new = new double *[n];
  for (unsigned int i = 0; i < n; i++)
    v_new[i] = new double[n];

  // copy the velocities to new fields
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      u_new[i][j] = field_u->mat[i][j];
      v_new[i][j] = field_v->mat[i][j];
    }
  }

  // parameters
  double **u_s = field_u->mat;
  double **v_s = field_v->mat;
  double dx = field_u->dx;
  double dy = field_v->dx;
  double rho = field_p->rho;
  unsigned int imax = field_u->imax;
  unsigned int imin = field_u->imin;
  unsigned int jmax = field_v->imax;
  unsigned int jmin = field_v->imin;

  // only updates interior nodes, correct boundary at the end
  // update u velocity
#pragma omp parallel for num_threads(threads_num)
  for (unsigned int j = jmin; j < jmax + 1; j++) {
    for (unsigned int i = imin + 1; i < imax + 1; i++) {
      u_new[i][j] = u_s[i][j] - dt / rho * (P[i][j] - P[i - 1][j]) / dx;
    }
  }

  // update v velocity
#pragma omp parallel for num_threads(threads_num)
  for (unsigned int j = jmin + 1; j < jmax + 1; j++) {
    for (unsigned int i = imin; i < imax + 1; i++) {
      v_new[i][j] = v_s[i][j] - dt / rho * (P[i][j] - P[i][j - 1]) / dx;
    }
  }

  // swith the pointer to new velocity field
  field_u->mat = u_new;
  field_v->mat = v_new;

  // deallocate
  for (unsigned int i = 0; i < n; i++) {
    delete[] u_s[i];
    delete[] v_s[i];
  }
  delete[] u_s;
  delete[] v_s;
}

void cfd_controller_omp(const double T, const double dt, Mesh *field_u,
                        Mesh *field_v, Mesh *field_p,
                        const size_t threads_num) {
  // number of iteration
  const unsigned int nt = (unsigned int)T / dt;

  // elasped duration
  float duration1 = 0, duration2 = 0, duration3 = 0;

  // start iteration
  for (unsigned int t = 0; t < nt; t++) {
    // start clocking
    start = std::chrono::high_resolution_clock::now();

    // boundary condition
    boundary_corr(field_u, field_v);

    // velocity discretization
    velocity_predict_u(field_u, field_v, dt, threads_num);
    velocity_predict_v(field_u, field_v, dt, threads_num);

    // stop clocking
    end = std::chrono::high_resolution_clock::now();

    // calculate elapsed time
    duration_sec =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);

    // update duration
    duration1 += duration_sec.count();

    // start clocking
    start = std::chrono::high_resolution_clock::now();

    // allocate a buffer for pressure field
    Mesh *temp = new Mesh(field_p->n, field_p->n);

    // pressure poisson
    pressure_poisson_matrix(field_u, field_v, dt, temp, threads_num);

    // update pressure
    for (unsigned int i = 0; i < temp->n; i++) {
      for (unsigned int j = 0; j < temp->n; j++) {
        field_p->mat[i][j] += temp->mat[i][j];
      }
    }

    // stop clocking
    end = std::chrono::high_resolution_clock::now();

    // calculate the elapsed time
    duration_sec =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);

    // update duration
    duration2 += duration_sec.count();

    // start clocking
    start = std::chrono::high_resolution_clock::now();

    // correction step
    correct_step(field_u, field_v, dt, field_p, threads_num);

    // stop clocking
    end = std::chrono::high_resolution_clock::now();

    // calcuate the elapsed time
    duration_sec =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);

    // update duration
    duration3 += duration_sec.count();

    // deallocate
    delete temp;

    // print out the iteration
    std::cout << t + 1 << "/" << nt << "\n";
  }

  // print out the duration
  std::cout << duration1 << " "
            << duration1 / (duration1 + duration2 + duration3) << " "
            << duration2 << " "
            << duration2 / (duration1 + duration2 + duration3) << " "
            << duration3 << " "
            << duration3 / (duration1 + duration2 + duration3) << "\n";

  // write the durations to file
  std::ofstream output_file("omp_time.txt");
  output_file << duration1 << " "
              << duration1 / (duration1 + duration2 + duration3) << " "
              << duration2 << " "
              << duration2 / (duration1 + duration2 + duration3) << " "
              << duration3 << " "
              << duration3 / (duration1 + duration2 + duration3) << "\n";

  // close file
  output_file.close();
}

// int main() {
//   // space config
//   const unsigned int nx = 32;

//   // time config
//   const double T = 5;
//   const double dt = 0.01;

//   // mesh config
//   Mesh *field_u = new Mesh(nx + 2, nx + 2);
//   Mesh *field_v = new Mesh(nx + 2, nx + 2);
//   Mesh *field_p = new Mesh(nx + 1, nx + 1);

//   // cdf solver
//   cfd_controller_omp(T, dt, field_u, field_v, field_p, 10);

//   // output result
//   field_u->write("u.txt");

//   delete field_p;
//   delete field_u;
//   delete field_v;
//   return 0;
// }