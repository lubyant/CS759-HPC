#include <iostream>
void swap(float **ap, float **bp);
int main(){
    float *a = new float[4]{1,2,3,4};
    float *b = new float[4]{4,3,2,1};
    swap(&a, &b);
    std::cout << a[0] << std::endl;
    std::cout << b[0] << std::endl;
}
void swap(float **ap, float **bp){
    float *temp;
    temp = *ap;
    *ap = *bp;
    *bp = temp;
}