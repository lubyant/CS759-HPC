#ifndef CFD_OMP_HPP
#define CFD_OMP_HPP
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
void cfd_controller_omp(const double T, const double dt, Mesh *field_u,
                        Mesh *field_v, Mesh *field_p,
                        const size_t threads_num);
#endif