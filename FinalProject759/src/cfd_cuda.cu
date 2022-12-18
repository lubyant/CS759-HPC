#include "cfd_cuda.cuh"
#include "field.hpp"
#include "splinear_cuda.cuh"
#include <cuda.h>
#include <stdio.h>

// timer
cudaEvent_t start, end;

void velocity_predict(Mesh *field_u, Mesh *field_v, const double dt,
                      const uint block_dim);

void boundary_corr(Mesh *field_u, Mesh *field_v);

void create_rhs(const Mesh *field_u, const Mesh *field_v, const double dt,
                Mesh *field_r);

void pressure_poisson_matrix(const Mesh *field_u, const Mesh *field_v,
                             const double dt, Mesh *field_p,
                             const uint block_dim);

void correct_step(Mesh *field_u, Mesh *field_v, const double dt,
                  const Mesh *field_p);

void predict_helper(double **u, double **v, double **u_s, double **v_s,
                    const uint n, const double dt, const double nu,
                    const uint imin, const uint imax, const uint jmin,
                    const uint jmax, const double dxi, const double dyi,
                    const uint block_dim);

__global__ void u_predict_kernel(double *u, double *v, double *u_s,
                                 const uint n, const double dt, const double nu,
                                 const uint imin, const uint imax,
                                 const uint jmin, const uint jmax,
                                 const double dxi, const double dyi);

__global__ void v_predict_kernel(double *u, double *v, double *v_s,
                                 const uint n, const double dt, const double nu,
                                 const uint imin, const uint imax,
                                 const uint jmin, const uint jmax,
                                 const double dxi, const double dyi);

void boundary_corr(Mesh *field_u, Mesh *field_v) {
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

// correction step for u velocity
void velocity_predict(Mesh *field_u, Mesh *field_v, const double dt,
                      const uint block_dim) {
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
  double **v_s = new double *[n];
  for (unsigned int i = 0; i < n; i++) {
    u_s[i] = new double[n];
    v_s[i] = new double[n];
  }

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      u_s[i][j] = u[i][j];
      v_s[i][j] = v[i][j];
    }
  }
  double nu = field_u->nu;

  // discretization
  predict_helper(u, v, u_s, v_s, n, dt, nu, imin, imax, jmin, jmax, dxi, dyi,
                 block_dim);

  field_u->mat = u_s;
  field_v->mat = v_s;

  // deallocate
  for (unsigned int i = 0; i < n; i++) {
    delete[] u[i];
    delete[] v[i];
  }
  delete[] u;
  delete[] v;
}

void predict_helper(double **u, double **v, double **u_s, double **v_s,
                    const uint n, const double dt, const double nu,
                    const uint imin, const uint imax, const uint jmin,
                    const uint jmax, const double dxi, const double dyi,
                    const uint block_dim) {
  // host pointer
  double *h_u = new double[n * n];
  double *h_v = new double[n * n];
  double *h_us = new double[n * n];
  double *h_vs = new double[n * n];

  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      h_u[i * n + j] = u[i][j];
      h_v[i * n + j] = v[i][j];
      h_us[i * n + j] = u[i][j];
      h_vs[i * n + j] = v[i][j];
    }
  }

  // device pointer and allcoate
  double *d_u, *d_v, *d_us, *d_vs;
  cudaMalloc(&d_u, n * n * sizeof(double));
  cudaMalloc(&d_v, n * n * sizeof(double));
  cudaMalloc(&d_us, n * n * sizeof(double));
  cudaMalloc(&d_vs, n * n * sizeof(double));

  // copy the host to device
  cudaMemcpy(d_u, h_u, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_us, h_us, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vs, h_vs, n * n * sizeof(double), cudaMemcpyHostToDevice);

  // call kernel
  dim3 DimBlock(block_dim, block_dim);
  dim3 DimGrid((n + block_dim - 1) / block_dim,
               (n + block_dim - 1) / block_dim);

  u_predict_kernel<<<DimGrid, DimBlock>>>(d_u, d_v, d_us, n, dt, nu, imin, imax,
                                          jmin, jmax, dxi, dyi);

  v_predict_kernel<<<DimGrid, DimBlock>>>(d_u, d_v, d_vs, n, dt, nu, imin, imax,
                                          jmin, jmax, dxi, dyi);

  // copy the result
  cudaMemcpy(h_us, d_us, n * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vs, d_vs, n * n * sizeof(double), cudaMemcpyDeviceToHost);
  for (uint i = 0; i < n; i++) {
    for (uint j = 0; j < n; j++) {
      u_s[i][j] = h_us[i * n + j];
      v_s[i][j] = h_vs[i * n + j];
    }
  }

  // deallocate
  delete[] h_u;
  delete[] h_v;
  delete[] h_us;
  delete[] h_vs;
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_us);
  cudaFree(d_vs);
}

__global__ void u_predict_kernel(double *u, double *v, double *u_s,
                                 const uint n, const double dt, const double nu,
                                 const uint imin, const uint imax,
                                 const uint jmin, const uint jmax,
                                 const double dxi, const double dyi) {
  // thread id
  uint tidx = threadIdx.x;
  uint tidy = threadIdx.y;
  uint idx = blockIdx.x * blockDim.x + tidx;
  uint idy = blockIdx.y * blockDim.y + tidy;

  double v_cur = 0, lap_op = 0;
  if (idx >= imin + 1 && idx < jmax + 1 && idy >= jmin && idy < jmax + 1) {
    v_cur = (v[(idx - 1) * n + idy] + v[(idx - 1) * n + idy + 1] +
             v[idx * n + idy] + v[idx * n + idy + 1]) /
            4;

    lap_op =
        (u[(idx + 1) * n + idy] - 2 * u[idx * n + idy] +
         u[(idx - 1) * n + idy]) *
            dxi * dxi +
        (u[idx * n + idy + 1] - 2 * u[idx * n + idy] + u[idx * n + idy - 1]) *
            dyi * dyi;

    u_s[idx * n + idy] =
        u[idx * n + idy] +
        dt *
            (nu * lap_op -
             u[idx * n + idy] *
                 (u[(idx + 1) * n + idy] - u[(idx - 1) * n + idy]) * 0.5 * dxi -
             v_cur * (u[idx * n + idy + 1] - u[idx * n + idy - 1]) * 0.5 * dyi);
  }
}

__global__ void v_predict_kernel(double *u, double *v, double *v_s,
                                 const uint n, const double dt, const double nu,
                                 const uint imin, const uint imax,
                                 const uint jmin, const uint jmax,
                                 const double dxi, const double dyi) {
  // thread id
  uint tidx = threadIdx.x;
  uint tidy = threadIdx.y;
  uint idx = blockIdx.x * blockDim.x + tidx;
  uint idy = blockIdx.y * blockDim.y + tidy;

  double u_cur, lap_op;
  if (idx >= imin + 1 && idx <= imax && idy >= jmin && idy <= jmax) {

    u_cur = (u[(idx - 1) * n + idy] + u[(idx - 1) * n + idy + 1] +
             u[idx * n + idy] + u[idx * n + idy + 1]) /
            4;

    lap_op =
        (v[(idx + 1) * n + idy] - 2 * v[idx * n + idy] +
         v[(idx - 1) * n + idy]) *
            dxi * dxi +
        (v[idx * n + idy + 1] - 2 * v[idx * n + idy] + v[idx * n + idy - 1]) *
            dyi * dyi;

    v_s[idx * n + idy] =
        v[idx * n + idy] +
        dt * (nu * lap_op -
              u_cur * (v[(idx + 1) * n + idy] - v[(idx - 1) * n + idy]) * 0.5 *
                  dxi -
              v[idx * n + idy] * (v[idx * n + idy + 1] - v[idx * n + idy - 1]) *
                  0.5 * dyi);
  }
}

void create_rhs(const Mesh *field_u, const Mesh *field_v, const double dt,
                Mesh *field_r) {
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

  for (unsigned int j = jmin; j < jmax + 1; j++) {
    for (unsigned int i = imin; i < imax + 1; i++) {
      cur_r = (-rho / dt) * ((u_s[i + 1][j] - u_s[i][j]) * dxi +
                             (v_s[i][j + 1] - v_s[i][j]) * dyi);
      field_r->arr[n++] = cur_r;
    }
  }
}

// create the pressure poisson matrix
void pressure_poisson_matrix(const Mesh *field_u, const Mesh *field_v,
                             const double dt, Mesh *field_p,
                             const uint block_dim) {
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
  for (unsigned int i = 0; i < n * n; i++)
    mat[0][i] = 0;
  mat[0][0] = 1;

  create_rhs(field_u, field_v, dt, &rhs);

  // linear solver
  double *ans = new double[n * n];
  splinear_solver_cuda(mat, rhs.arr, n * n, ans, block_dim);

  unsigned int count = 0;
  for (unsigned int j = jmin; j < jmax + 1; j++) {
    for (unsigned int i = imin; i < imax + 1; i++) {
      field_p->mat[i][j] = ans[count++];
    }
  }

  delete[] ans;
}

void correct_step(Mesh *field_u, Mesh *field_v, const double dt,
                  const Mesh *field_p) {
  unsigned int n = field_u->n;
  double **P = field_p->mat;
  double **u_new = new double *[n];
  for (unsigned int i = 0; i < n; i++)
    u_new[i] = new double[n];

  double **v_new = new double *[n];
  for (unsigned int i = 0; i < n; i++)
    v_new[i] = new double[n];

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      u_new[i][j] = field_u->mat[i][j];
      v_new[i][j] = field_v->mat[i][j];
    }
  }

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
  for (unsigned int j = jmin; j < jmax + 1; j++) {
    for (unsigned int i = imin + 1; i < imax + 1; i++) {
      u_new[i][j] = u_s[i][j] - dt / rho * (P[i][j] - P[i - 1][j]) / dx;
    }
  }
  for (unsigned int j = jmin + 1; j < jmax + 1; j++) {
    for (unsigned int i = imin; i < imax + 1; i++) {
      v_new[i][j] = v_s[i][j] - dt / rho * (P[i][j] - P[i][j - 1]) / dy;
    }
  }

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

void cfd_controller_cuda(const double T, const double dt, Mesh *field_u,
                         Mesh *field_v, Mesh *field_p,
                         const unsigned int block_dim) {
  // number of iteration
  const unsigned int nt = (unsigned int)T / dt;

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // initialize the duration
  float duration1 = 0, duration2 = 0, duration3 = 0, ms = 0;

  // start iteration
  for (unsigned int t = 0; t < nt; t++) {
    // start clocking
    cudaEventRecord(start);

    // boundary condition
    boundary_corr(field_u, field_v);

    // velocity discretization
    velocity_predict(field_u, field_v, dt, block_dim);

    // stop clocking
    cudaEventRecord(end);

    // synchronize
    cudaEventSynchronize(end);

    // calculate the elapsed time
    cudaEventElapsedTime(&ms, start, end);

    // update the duration
    duration1 += ms;

    // start clocking
    cudaEventRecord(start);

    // pressure poisson
    Mesh *temp = new Mesh(field_p->n, field_p->n);
    pressure_poisson_matrix(field_u, field_v, dt, temp, block_dim);

    // update pressure
    for (unsigned int i = 0; i < temp->n; i++) {
      for (unsigned int j = 0; j < temp->n; j++) {
        field_p->mat[i][j] += temp->mat[i][j];
      }
    }

    // stop clocking
    cudaEventRecord(end);

    // synchronize
    cudaEventSynchronize(end);

    // calculate elapsed time
    cudaEventElapsedTime(&ms, start, end);

    // update duration
    duration2 += ms;

    // start clocking
    cudaEventRecord(start);

    // correction step
    correct_step(field_u, field_v, dt, field_p);

    // deallocate
    delete temp;
    printf("%d/%d\n", t + 1, nt);

    // stop clocking
    cudaEventRecord(end);

    // synchronize
    cudaEventSynchronize(end);

    // calculate elapsed time
    cudaEventElapsedTime(&ms, start, end);

    // update duration
    duration3 += ms;
  }
  // print out the results
  printf("%f %f %f %f %f %f\n", duration1,
         duration1 / (duration1 + duration2 + duration3), duration2,
         duration1 / (duration1 + duration2 + duration3), duration3,
         duration1 / (duration1 + duration2 + duration3));

  // write the results to a file
  FILE *fp;
  fp = fopen("cuda_time.txt", "w+");
  fprintf(fp, "%f %f %f %f %f %f\n", duration1,
          duration1 / (duration1 + duration2 + duration3), duration2,
          duration1 / (duration1 + duration2 + duration3), duration3,
          duration1 / (duration1 + duration2 + duration3));
  fclose(fp);
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
//   cfd_controller_cuda(T, dt, field_u, field_v, field_p);

//   // output result
//   field_u->write("u_cuda.txt");

//   delete field_p;
//   delete field_u;
//   delete field_v;
//   return 0;
// }