#include "count.cuh"
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

void count(const thrust::device_vector<int> &d_in,
           thrust::device_vector<int> &values,
           thrust::device_vector<int> &counts) {
  // length of input array
  int len = d_in.size();

  // child array for count the distinct element
  thrust::device_vector<int> count_vec(len, 1);

  // copy the array and sort
  thrust::device_vector<int> d_in_copy(len);
  thrust::copy(d_in.begin(), d_in.end(), d_in_copy.begin());
  thrust::sort(d_in_copy.begin(), d_in_copy.end());

  // reduce the array by key
  thrust::pair<thrust::device_vector<int>::iterator,
               thrust::device_vector<int>::iterator>
      new_end;
  new_end =
      thrust::reduce_by_key(thrust::device, d_in_copy.begin(), d_in_copy.end(),
                            count_vec.begin(), values.begin(), counts.begin());

  // write the result into values and counts
  int reduced_len = new_end.first - values.begin();
  values.erase(values.begin() + reduced_len, values.end());
  counts.erase(counts.begin() + reduced_len, counts.end());
}

// int main(){
//     uint n = 20;
//     thrust::host_vector<int> hv(n);
//     for(uint i=0; i<n; i++){
//         hv[i] = (5*(float)rand()/(float)RAND_MAX);
//     }

//     thrust::device_vector<int> dv(n);
//     thrust::copy(hv.begin(), hv.end(), dv.begin());
//     thrust::device_vector<int> values(n), counts(n);
//     count(dv, values, counts);
//     thrust::host_vector<int> values_h = values;
//     thrust::host_vector<int> counts_h = counts;
//     for(uint i=0; i<values.size(); i++){
//         printf("%d %d \n", values_h[i], counts_h[i]);
//     }
//     return 0;
// }
