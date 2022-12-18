#include "optimize.h"

int vec_length(vec *);

void optimize1(vec *v, data_t *dest) {
  int len = vec_length(v);
  data_t *d = get_vec_start(v);
  data_t temp = IDENT;
  for (int i = 0; i < len; i++) {
    temp = temp OP d[i];
  }
  *dest = temp;
}
void optimize2(vec *v, data_t *dest) {
  int len = vec_length(v);
  int limit = len - 1;
  data_t *d = get_vec_start(v);
  data_t x = IDENT;
  int i;

  // reduce 2 elements at a time
  for (i = 0; i < limit; i += 2) {
    x = (x OP d[i])OP d[i + 1];
  }

  // finish any remaining elements
  for (; i < len; i++) {
    x = x OP d[i];
  }

  *dest = x;
}
void optimize3(vec *v, data_t *dest) {
  int len = vec_length(v);
  int limit = len - 1;
  data_t *d = get_vec_start(v);
  data_t x = IDENT;
  int i;

  // reduce 2 elements at a time
  for (i = 0; i < limit; i += 2) {
    x = x OP(d[i] OP d[i + 1]);
  }

  // finish any remaining elements
  for (; i < len; i++) {
    x = x OP d[i];
  }
  *dest = x;
}
void optimize4(vec *v, data_t *dest) {
  long len = (long)vec_length(v);
  long limit = len - 1;
  data_t *d = get_vec_start(v);
  data_t x0 = IDENT;
  data_t x1 = IDENT;
  long i;

  // reduce 2 elements at a time
  for (i = 0; i < limit; i += 2) {
    x0 = x0 OP d[i];
    x1 = x1 OP d[i + 1];
  }

  // finish any remaining elements
  for (; i < len; i++) {
    x0 = x0 OP d[i];
  }

  *dest = x0 OP x1;
}
void optimize5(vec *v, data_t *dest) {
  int len = vec_length(v);
  int limit = len - 1;
  data_t *d = get_vec_start(v);

  // L = 3
  data_t x0 = IDENT;
  data_t x1 = IDENT;
  data_t x2 = IDENT;
  int i;

  // reduce 3 elements at a time
  for (i = 0; i < limit; i += 3) {
    x0 = x0 OP d[i];
    x1 = x1 OP d[i + 1];
    x2 = x2 OP d[i + 2];
  }

  // finish any remaining elements
  for (; i < len; i++) {
    x0 = x0 OP d[i];
  }
  *dest = x0 OP x1 OP x2;
}

int vec_length(vec *v) { return v->len; }

data_t *get_vec_start(vec *v) { return v->data; }

// int main(int argc, char *argv[]){
//     size_t n = 10;

//     // initialize vec
//     vec *v = new vec(n);
//     v->data = new data_t[n];
//     for(size_t i=0; i<n; i++){
//         v->data[i] = (data_t)1;
//     }
//     data_t *dist = new data_t;

//     // optimization 1
//     optimize1(v, dist);

//     // deallocate
//     delete[] v->data;
//     delete[] v;
//     delete[] dist;
//     return 0;
// }