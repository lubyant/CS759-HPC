#ifndef CFD
#define CFD

#include "field.hpp"

/**
 * @brief controller for CFD solver without parallel accleration
 * 
 * @param T total time for assimulation (second)
 * @param dt time step
 * @param field_u u velocity field
 * @param field_v v velocity field
 * @param field_p pressure field
 */
void cfd_controller(const double T, const double dt, Mesh *field_u,
                    Mesh *field_v, Mesh *field_p);


#endif
