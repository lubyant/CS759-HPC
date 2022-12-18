# ME/CS/ECE759 High Performance Computing: Final Project
## Environment:
OS: Ubuntu 20.04\
C++ compiler: gcc 9.4 or newer\
CUDA complier: nvcc 10.1 or newer

## Compile the code:
Clone the repo and go to root directory. Build the program as following:
```
cd FinalProject759
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
After building the codes, you shall have executable file "CS759_project" as well as three share libraries built in the folder "libcfd.so", "libcfd_omp.so", and "libcfd_cuda.so". Here are the explanations for these files.

CS759_project: main entrance of the program \
libcfd.so: subroutine for sequential computation \
libcfd_omp.so: subroutine for OpenMP \
libcfd_cuda.so: subroutine for CUDA 
## Argument for the program


Runing the program by following argument
```
./CS759_project arg1 arg2 arg3 arg4 arg5
```

where:\
arg1: size of fluid field \
arg2: time step in second \
arg3: number of threads for OpenMP \
arg4: size of threads block for CUDA \
arg5: running mode (0 - all, 1 - no parallel, 2 - OpenMp, 3 - CUDA)

### Example
```
# running non-parallel, OpenMP (10 threads), and CUDA (16*16) threads block
# for a 32*32 1-meter grid at dt=0.01s simulation
./CS759_project 32 0.01 10 16 0 
```

