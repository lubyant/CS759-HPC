#include <iostream>
#include <chrono>

typedef unsigned int uint;
using namespace std;
void stencil(float *image, float *mask, float *output, uint n, uint R){
    for(uint i=0; i<n; i++){
        float value = 0;
        for(uint j=0; j<2*R+1; j++){
            int m = mask[j];
            float img;
            if(i-R+j < 0 || i-R+j > n-1)
                img = 1;
            else
                img = image[i-R+j];
            value += m*img;
        }
        output[i] = value;
    }
}


int main(int argc, char *argv[]){
    uint n, R;
    n = atoi(argv[1]);
    R = atoi(argv[2]);

    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point end;
    chrono::duration<double, milli> duration_sec;

    // uint n=1024, R=128, threads_per_block=1024;
    // generate random image and mask
    float *image = new float[n];
    float *mask = new float[2*R+1];
    float *output = new float[n]{0};
    srand(time(NULL));
    for(uint i=0; i<n; i++){    
        image[i] = 2*(float)rand()/((float)RAND_MAX)-1;
    }
    for(uint i=0; i<2*R+1; i++){
        mask[i] = 2*(float)rand()/((float)RAND_MAX)-1;
    }
    start = chrono::high_resolution_clock::now();
    stencil(image, mask, output, n, R);
    end = chrono::high_resolution_clock::now();
    duration_sec =
      chrono::duration_cast<chrono::duration<double, milli>>(end - start);
    std::cout<< duration_sec.count() << "\n";
    return 0;
}