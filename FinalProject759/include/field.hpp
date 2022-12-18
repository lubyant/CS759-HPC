#ifndef FIELD_HPP
#define FIELD_HPP
#include <string>


// class for physical setting
class Settings {
public:
  // space setting
  static constexpr double Lx = 1;
  static constexpr double Ly = 1;


  // physical parameter
  static constexpr double Re = 100;
  static constexpr double u_top = 1;
  static constexpr double gamma = 1;
  static constexpr double sigma = 2.5;
  static constexpr double nu = 0.01;
  static constexpr double rho = u_top * Lx / Re;

  unsigned int n;

  // constructor
  Settings(unsigned int);

  // default constructor
  Settings() = default;

  // destructor
  ~Settings() = default;
};

// Mesh class for velocity and pressure fields
class Mesh : public Settings {
public:
  // fields
  unsigned int dxi;
  unsigned int imin = 1;
  unsigned int imax;
  double dx;
  double *arr = nullptr;  // squash array length n*n
  double **mat = nullptr; // 2d matrix n*n

  // constructor for arr
  Mesh(unsigned int n);

  // constructor for matrix
  Mesh(unsigned int n1, unsigned int n2);

  // convert matrix to arr
  void mat2arr();

  // convert arr to matrix
  void arr2mat();

  // print out the matrix
  void print();

  // clear the array
  void clear();

  // write the mesh into files
  void write(std::string output_path);

  // default constructor
  Mesh();

  // destructor
  ~Mesh();
};

#endif