#include "field.hpp"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string.h>

Settings::Settings(unsigned int n) : n(n){};

// create a squash 1d array
Mesh::Mesh(unsigned int n) : Settings{n} {
  this->arr = new double[n * n];
  for (unsigned int i = 0; i < n * n; i++) {
    arr[i] = 0;
  }
  this->dx = (double)Settings::Lx / (n - 2);
  this->dxi = (unsigned int)1 / dx;
  this->imax = imin + n - 3;
}

// create a 2d matrix
Mesh::Mesh(unsigned int n1, unsigned int n2) : Settings(n1) {

  // assign a 2d pointer
  this->mat = new double *[n1];
  for (unsigned int i = 0; i < n2; i++) {
    mat[i] = new double[n2];
  }

  // init as 1
  for (unsigned int i = 0; i < n1; i++) {
    for (unsigned int j = 0; j < n2; j++) {
      this->mat[i][j] = 0;
    }
  }

  this->dx = (double)Settings::Lx / (n - 2);
  this->dxi = (unsigned int)1 / dx;
  this->imax = imin + n - 3;
}

// 1d to 2d
void Mesh::arr2mat() {
  assert(arr != nullptr);
  this->mat = new double *[n];
  for (unsigned int i = 0; i < n; i++) {
    mat[i] = new double[n];
  }
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      mat[i][j] = arr[i * n + j];
    }
  }

  // deallocate arr to save storage
  delete[] arr;
}

// 2d to 1d
void Mesh::mat2arr() {
  assert(mat != nullptr);
  this->arr = new double[n * n];
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      arr[i * n + j] = mat[i][j];
    }
  }

  // deallocate mat to save storage
  delete[] mat;
}

Mesh::Mesh() : Settings() {}

Mesh::~Mesh() {
  if (this->arr != nullptr)
    delete[] arr;

  if (this->mat != nullptr)
    for (unsigned int i = 0; i < n; i++) {
      delete[] mat[i];
    }
}

void Mesh::print() {
  unsigned int n = this->n;
  if (this->mat != nullptr) {
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int j = 0; j < n; j++) {
        std::cout << this->mat[i][j] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  } else if (this->arr != nullptr) {
    for (unsigned int i = 0; i < n * n; i++)
      std::cout << this->arr[i] << " ";
    std::cout << "\n";
  } else {
    std::cout << "Nothing to print\n";
  }
}

void Mesh::write(std::string output_path) {
  // file class
  std::ofstream output_file(output_path);

  if (this->mat == nullptr && this->arr != nullptr) {
    this->arr2mat();
  }

  if (this->mat != nullptr) {
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int j = 0; j < n; j++) {
        output_file << this->mat[i][j] << " ";
      }
      output_file << "\n";
    }
    output_file << "\n";
  }

  // close file
  output_file.close();
}

void Mesh::clear() {
  if (this->mat == nullptr && this->arr != nullptr) {
    for (unsigned int i = 0; i < n * n; i++) {
      arr[i] = 0;
    }
    return;
  }

  if (this->mat != nullptr) {
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int j = 0; j < n; j++) {
        this->mat[i][j] = 0;
      }
    }
  }
}
// int main() {
//   Mesh *field = new Mesh(10, 10);
//   // std::cout << field->mat[0][0] << std::endl;
//   field->print();
//   field->write("output.txt");
//   delete field;
//   return 0;
// }