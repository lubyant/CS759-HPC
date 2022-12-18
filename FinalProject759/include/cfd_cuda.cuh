#ifndef CFD_CUDA_CUH
#define CFD_CUDA_CUH
#include "field.hpp"


/**
 * @brief controller for cfd solver with OpenMP accleration
 *
 * @param T total time of assimulation (second)
 * @param dt time step
 * @param field_u u velocity field
 * @param field_v v velocity field
 * @param field_p pressure field
 * @param threads_num number of threads for computing
 */
void cfd_controller_cuda(const double T, const double dt, Mesh *field_u,
                         Mesh *field_v, Mesh *field_p, const unsigned int block_dim);
#endif