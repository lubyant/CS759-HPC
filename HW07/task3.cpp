#include <cstdio>
#include <omp.h>
// function for calculate factorial
int fac(int n) {
  int ans = 1;
  for (int i = 1; i <= n; i++) {
    ans *= i;
  }
  return ans;
}
int main() {
  // set four threads
  printf("Number of threads:  4\n");

  // print out the thread id
#pragma omp parallel num_threads(4)
  { printf("I am thread No.  %d\n", omp_get_thread_num()); }

// parallelize with for clause
#pragma omp parallel num_threads(4)
  {
#pragma omp for
    for (int i = 1; i <= 8; i++) {
      printf("%d!=%d\n", i, fac(i));
    }
  }
  return 0;
}