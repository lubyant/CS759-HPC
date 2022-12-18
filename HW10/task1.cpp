#include "optimize.h"
#include <chrono>
#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
  size_t n = (size_t)atoi(argv[1]);

  // timer
  chrono::high_resolution_clock::time_point start;
  chrono::high_resolution_clock::time_point end;
  chrono::duration<double, std::milli> duration_sec;

  // initialize vec
  vec v(n);
  v.data = new data_t[n];
  for (size_t i = 0; i < n; i++) {
    v.data[i] = (data_t)1;
  }
  data_t *dist = new data_t{0};

  // optimization 1
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++)
    optimize1(&v, dist);
  end = std::chrono::high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);
  cout << *dist << "\n" << duration_sec.count() / 10 << "\n";

  // optimization 2
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++)
    optimize2(&v, dist);
  end = std::chrono::high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);
  cout << *dist << "\n" << duration_sec.count() / 10 << "\n";

  // optimization 3
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++)
    optimize3(&v, dist);
  end = std::chrono::high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);
  cout << *dist << "\n" << duration_sec.count() / 10 << "\n";

  // optimization 4
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++)
    optimize4(&v, dist);
  end = std::chrono::high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);
  cout << *dist << "\n" << duration_sec.count() / 10 << "\n";

  // optimization 5
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; i++)
    optimize5(&v, dist);
  end = std::chrono::high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);
  cout << *dist << "\n" << duration_sec.count() / 10 << "\n";

  // deallocate
  delete[] v.data;
  return 0;
}