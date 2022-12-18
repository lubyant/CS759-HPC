#include "cfd.hpp"
#include "cfd_cuda.cuh"
#include "cfd_omp.hpp"
#include <iostream>

const double T = 5;
void cfd_seq(const unsigned int nx, const double dt);
void cfd_omp(const unsigned int nx, const double dt,
             const unsigned int threads_num);
void cfd_cuda(const unsigned int nx, const double dt,
              const unsigned int block_dim);

int main(int argc, char *argv[]) {
  // fisrt input argument: field dimension
  unsigned int nx = (unsigned int)atoi(argv[1]);

  // second input argument: time step
  double dt = (double)atof(argv[2]);

  // third input argument: number of threads for OpenMP
  unsigned int threads_num = (unsigned int)atoi(argv[3]);

  // forth input argument: number of threads for CUDA
  unsigned int block_dim = (unsigned int)atoi(argv[4]);

  // fifth input argument: runing one pack
  // 0 - all, 1 - seq, 2 - omp, 3 - cuda
  unsigned int pattern = (unsigned int)atoi(argv[5]);

  // seqential computation
  if (pattern == 1 || pattern == 0)
    cfd_seq(nx, dt);

  // parallel computation on OpenMP
  if (pattern == 2 || pattern == 0)
    cfd_omp(nx, dt, threads_num);

  // parallel computation on CUDA
  if (pattern == 3 || pattern == 0)
    cfd_cuda(nx, dt, block_dim);

  return 0;
}

void cfd_seq(const unsigned int nx, const double dt) {
  // mesh config
  Mesh *field_u = new Mesh(nx + 2, nx + 2);
  Mesh *field_v = new Mesh(nx + 2, nx + 2);
  Mesh *field_p = new Mesh(nx + 1, nx + 1);

  // sequential computing start
  std::cout << "Computing start: no parallel\n";

  // call cfd solver
  cfd_controller(T, dt, field_u, field_v, field_p);

  // output the result
  field_u->write("u_seq.txt");

  // deallocate
  delete field_p;
  delete field_u;
  delete field_v;
}

void cfd_omp(const unsigned int nx, const double dt,
             const unsigned int threads_num) {
  // mesh config
  Mesh *field_u = new Mesh(nx + 2, nx + 2);
  Mesh *field_v = new Mesh(nx + 2, nx + 2);
  Mesh *field_p = new Mesh(nx + 1, nx + 1);

  // parallel computing start
  std::cout << "Parallel computing start: OpenMP\n";

  // call cfd solver
  cfd_controller_omp(T, dt, field_u, field_v, field_p, threads_num);

  // output the result
  field_u->write("u_omp.txt");

  // deallocate
  delete field_p;
  delete field_u;
  delete field_v;
}

void cfd_cuda(const unsigned int nx, const double dt,
              const unsigned int threads_num) {
  // mesh config
  Mesh *field_u = new Mesh(nx + 2, nx + 2);
  Mesh *field_v = new Mesh(nx + 2, nx + 2);
  Mesh *field_p = new Mesh(nx + 1, nx + 1);

  // parallel computing start
  std::cout << "Parallel computing start: CUDA\n";

  // call cfd solver
  cfd_controller_cuda(T, dt, field_u, field_v, field_p, threads_num);

  // output the result
  field_u->write("u_cuda.txt");

  // deallocate
  delete field_p;
  delete field_u;
  delete field_v;
}