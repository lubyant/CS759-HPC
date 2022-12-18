#include <iostream>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
  char *input = argv[1];
  int N = atoi(input);
  for (int i = 0; i <= N; i++) {
    printf("%d ", i);
  }
  printf("\n");
  for (int i = N; i >= 0; i--) {
    std::cout << i << " ";
  }
  std::cout << "\n";
}
