#include "stdio.h"
#include "stencil.cuh"
typedef unsigned int uint;
int main(int argc, char *argv[]) {
  uint n, R, threads_per_block; // arr len, radius, threads
  n = atoi(argv[1]);
  R = atoi(argv[2]);
  threads_per_block = atoi(argv[3]);

  // generate random image and mask
  float *image = new float[n];
  float *mask = new float[2 * R + 1];
  float *output = new float[n]{0};
  srand(time(NULL));
  for (uint i = 0; i < n; i++) {
    image[i] = 2 * (float)rand() / ((float)RAND_MAX) - 1;
  }
  for (uint i = 0; i < 2 * R + 1; i++) {
    mask[i] = 2 * (float)rand() / ((float)RAND_MAX) - 1;
  }

  // call kernel function
  stencil(image, mask, output, n, R, threads_per_block);

  // deallocate
  delete[] image;
  delete[] mask;
  delete[] output;
  mask = nullptr;
  image = nullptr;
  output = nullptr;

  return 0;
}


/**
 * @brief some test case
 * 
 */
// #include "stencil.cuh"
// #include "stdio.h"

// int main(){
//     float *image = new float[16]{0,1,2,3,4,5,6,7,8,9,10,11,10,10,10,15};
//     float *mask = new float[9]{0,1,2,3,4,5,6, 7, 8};
//     float *output = new float[16]{0};  
//     // call kernel function
//     stencil(image, mask, output, 16, 4, 512);

//     // print the result
//     for(int i=0; i<16; i++){
//         printf("%f\n", output[i]);
//     }
//     return 0;
// }    
