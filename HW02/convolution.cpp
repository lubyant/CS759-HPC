#include "convolution.h"
#include <iostream>

class Cartesian
{
public:
    int i, j, x, n;
    void convert1()
    {
        i = x / n;
        j = x % n;
    }
    void convert2()
    {
        x = i * n + j;
    }
    Cartesian() = default;

    Cartesian(int x, int n)
    {
        this->x = x;
        this->n = n;
        this->convert1();
    }
    Cartesian(int i, int j, int n)
    {
        this->i = i;
        this->j = j;
        this->n = n;
        this->convert2();
    }
};

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    using namespace std;
    float entry, image_e;
    int image_i, image_j, mask_i, mask_j;


    for (size_t i = 0; i < n * n; i++)
    {
        entry = 0;
        Cartesian cord_img(i, n);
        image_i = cord_img.i;
        image_j = cord_img.j;
        for (size_t j = 0; j < m * m; j++)
        {
            Cartesian cord_mask(j, m);
            mask_i = cord_mask.i;
            mask_j = cord_mask.j;
            if( (image_i + mask_i - (m - 1) / 2 < 0
            && image_j + mask_j - (m - 1) / 2 < 0)
            || (image_i + mask_i - (m - 1) / 2 > n - 1
            && image_j + mask_j - (m - 1) / 2 > n - 1)){
                image_e = 0;
            }
            else if (image_i + mask_i - (m - 1) / 2 < 0 
            || image_i + mask_i - (m - 1) / 2 > n - 1 
            || image_j + mask_j - (m - 1) / 2 < 0 
            || image_j + mask_j - (m - 1) / 2 > n - 1)
            {
                image_e = 1;
            } else {
                Cartesian cor(image_i + mask_i - (m - 1) / 2, image_j + mask_j - (m - 1) / 2, n);
                image_e = image[cor.x];
            }
            entry += mask[j] * image_e;
        }
        output[i] = entry;
    }
}

// int main()
// {
//     float image[] = {
//         1,
//         3,
//         4,
//         8,
//         6,
//         5,
//         2,
//         4,
//         3,
//         4,
//         6,
//         8,
//         1,
//         4,
//         5,
//         2,
//     };
//     float mask[] = {
//         0,
//         0,
//         1,
//         0,
//         1,
//         0,
//         1,
//         0,
//         0,
//     };
//     float output[16];
//     convolve(image, output, 4, mask, 3);
//     for(int i=0; i<16; i++){
//         std::cout<<output[i]<<"\n";
//     }
//     return 0;
// }