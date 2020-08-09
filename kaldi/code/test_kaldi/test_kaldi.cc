#include "matrix/kaldi-matrix.h"

int main() {
  kaldi::Matrix<float> m(2, 3);
  m.Write(std::cout, false);
  return 0;
}
