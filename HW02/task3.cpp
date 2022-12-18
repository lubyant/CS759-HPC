#include "matmul.h"
#include <iostream>
#include <vector>
#include <chrono>
int main(){
    using namespace std;
    int n = 1024;

    cout<< n << "\n";

    double *A = new double[n*n];
    double *B = new double[n*n];
    double *C = new double[n*n];

    vector<double> A_v(n*n);
    vector<double> B_v(n*n);
    
    
    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point end;
    chrono::duration<double, milli> duration_sec;

    for(int i=0; i<n*n; i++){
        A[i] = 10*((double)rand()) / (double)RAND_MAX;
        B[i] = 10*((double)rand()) / (double)RAND_MAX;
        A_v[i] = A[i];
        B_v[i] = B[i];
    }

    // test mmul1
    start = chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    end = chrono::high_resolution_clock::now(); 
    duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
    cout << duration_sec.count() << "\n" << C[n*n-1] << "\n";

    // test mmul2
    start = chrono::high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = chrono::high_resolution_clock::now(); 
    duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
    cout << duration_sec.count() << "\n" << C[n*n-1] << "\n";

    // test mmul3
    start = chrono::high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = chrono::high_resolution_clock::now(); 
    duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
    cout << duration_sec.count() << "\n" << C[n*n-1] << "\n";

    // test mmul4
    start = chrono::high_resolution_clock::now();
    mmul4(A_v, B_v, C, n);
    end = chrono::high_resolution_clock::now(); 
    duration_sec = chrono::duration_cast<chrono::duration<double, milli>>(end - start);
    cout << duration_sec.count() << "\n" << C[n*n-1] << "\n";

    return 0;
}