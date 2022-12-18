#include "mpi.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  using namespace std;

  // parameter
  int n = (int)atoi(argv[1]);
  int my_rank, source, dest, tag = 0;

  // messeage array send and recieve
  float *message_s = new float[n];
  float *message_r = new float[n];

  // container for the elapsed time that going to send and recieve
  float *duration_s = new float;
  float *duration_r = new float;

  // create the array message
  for (int i = 0; i < n; i++) {
    message_s[i] = 1;
    message_r[i] = 0;
  }

  // status
  MPI_Status status;

  // timers
  chrono::high_resolution_clock::time_point start1, start2;
  chrono::high_resolution_clock::time_point end1, end2;
  chrono::duration<double, std::milli> duration_sec1, duration_sec2;

  // init the MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) { // process 0
    start1 = chrono::high_resolution_clock::now();

    // send to process 1
    dest = 1;
    MPI_Send(message_s, n, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);

    // recieve from process 1
    source = 1;
    MPI_Recv(message_r, n, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);

    end1 = chrono::high_resolution_clock::now();
    duration_sec1 =
        chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end1 - start1);

    // recieve the elapsed time from process 1
    MPI_Recv(duration_r, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
    printf("%f\n", duration_r + duration_sec1.count());

  } else if (my_rank == 1) { // process 1
    start2 = chrono::high_resolution_clock::now();

    // recieve from process 0
    source = 0;
    MPI_Recv(message_r, n, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);

    // send to process 1
    dest = 0;
    MPI_Send(message_s, n, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
    end2 = chrono::high_resolution_clock::now();

    duration_sec2 =
        chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end2 - start2);

    // send the elapsed time to process 1
    duration_s = &duration_sec2.count();
    MPI_Send(duration_s, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD);
  } else {
    // in case too many threads
    printf("Wrong argument");
  }

  // deallocate
  delete[] message_s;
  delete[] message_r;
  delete duration_r;
  delete duration_s;

  // close MPI
  MPI_Finalize();
  return 0;
}