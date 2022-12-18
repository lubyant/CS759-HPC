#include "msort.h"
#include <bits/stdc++.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

void merge(int *arr1, const std::size_t n1, int *arr2, const std::size_t n2);
void merge_sort(int *arr, const std::size_t left, const std::size_t right,
                const std::size_t threshold);
void merge_sort_seq(int *arr, const std::size_t left, const std::size_t right);

/**
 * @brief merge sort interface
 *
 * @param arr input array
 * @param n length of array
 * @param threshold threshold for serial sort
 */
void msort(int *arr, const std::size_t n, const std::size_t threshold) {
  if (n > threshold) { // parallel merge sort
#pragma omp parallel
#pragma omp single nowait
    merge_sort(arr, 0, n - 1, threshold);
  } else { // sequential merge sort
    merge_sort_seq(arr, 0, n - 1);
  }
}

// implementation of parallel merge sort
void merge_sort(int *arr, const std::size_t left, const std::size_t right,
                const std::size_t threshold) {
  // termination when cannot divide
  if (right - left < 1)
    return;

  // length of total
  std::size_t n = right - left + 1;

  // length of left half
  std::size_t n1 = n / 2;

  // length of right half
  std::size_t n2 = n - n1;

  if (n > threshold) {
    // parallel block for left half
#pragma omp task shared(arr) if (n1 >= threshold)
    merge_sort(arr, left, left + n1 - 1, threshold);

    // parallel block for right half
#pragma omp task shared(arr) if (n2 >= threshold)
    merge_sort(arr + n1, left + n1, right, threshold);

    // synchronization then merge
#pragma omp taskwait
    merge(arr, n1, arr + n1, n2);
  } else {
    merge_sort_seq(arr, left, right);
  }
}

// implementation of merge
void merge(int *arr1, const std::size_t n1, int *arr2, const std::size_t n2) {
  // temp arr for storing the result temporarily
  int *temp = new int[n1 + n2];

  // ptr1 to arr1 ptr2 to arr2
  volatile std::size_t ptr1 = 0;
  volatile std::size_t ptr2 = 0;

  // compare arr1 and arr2 first element and put the small one in temp
  while (ptr1 < n1 && ptr2 < n2) {
    if (arr1[ptr1] < arr2[ptr2]) {
      temp[ptr2 + ptr1] = arr1[ptr1];
      ptr1++;
    } else {
      temp[ptr1 + ptr2] = arr2[ptr2];
      ptr2++;
    }
    // temp[ptr1 + ptr2 - 1] =
    //     arr1[ptr1] < arr2[ptr2] ? arr1[ptr1++] : arr2[ptr2++];
  }

  // put arr1 in temp
  while (ptr1 < n1) {
    temp[ptr1 + ptr2] = arr1[ptr1];
    ptr1++;
  }

  // put arr2 in temp
  while (ptr2 < n2) {
    temp[ptr1 + ptr2] = arr2[ptr2];
    ptr2++;
  }

  // copy result of temp into arr1
  memcpy(arr1, temp, sizeof(int) * (n1 + n2));

  // deallocate
  delete[] temp;
}

/**
  Actually, I don't this sequential implementation is necessary
  because (#pragma omp if) clause would surpress the parallel task when
  size is overly small. Although, I found it is no different in result,
  I put implementation here because I see some posts in Piazza suggests to do
  so.
 */
void merge_sort_seq(int *arr, const std::size_t left, const std::size_t right) {
  // termination when cannot divide
  if (right - left < 1)
    return;

  // length of total
  std::size_t n = right - left + 1;

  // length of left half
  std::size_t n1 = n / 2;

  // length of right half
  std::size_t n2 = n - n1;

  // parallel block for left half
  merge_sort_seq(arr, left, left + n1 - 1);

  // parallel block for right half
  merge_sort_seq(arr + n1, left + n1, right);

  // synchronization then merge
  merge(arr, n1, arr + n1, n2);
}
// int main() {
//   int *arr = new int[10]{3, 1, 3, 6, 9, -1, 4, 0, 5, -8};
//   omp_set_num_threads(4);
//   msort(arr, 10, 0);
//   for (int i = 0; i < 10; i++)
//     printf("%d ", arr[i]);
//   printf("\n");
//   return 0;
// }