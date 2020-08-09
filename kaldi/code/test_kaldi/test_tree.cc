#include "tree/context-dep.h"
#include "util/kaldi-io.h"

using namespace kaldi;

void test() {
  std::vector<int> phones = {1, 2};
  std::vector<int> phones2num_pdf_classes = {-1, 3, 3};
  ContextDependency* ctx_dep =
      MonophoneContextDependency(phones, phones2num_pdf_classes);
  ctx_dep->Write(std::cout, false);
  /*
   ContextDependency 1 0 ToPdf
   TE 0 3
    ( NULL
      TE -1 3 ( CE 0 CE 1 CE 2)
      TE -1 3 ( CE 3 CE 4 CE 5)
    )
   */
  Output o("tree", false);
  ctx_dep->Write(o.Stream(), false);
  std::cout << "num pdfs: " << ctx_dep->NumPdfs() << "\n";  // num pdfs: 6
  o.Close();
  // draw-tree phones.txt tree | dot -Tpdf > tree.pdf
  // where phones.txt is
  // eps 0
  // A 1
  // B 2
}

void test2() {
  std::vector<std::vector<int>> shared_phones = {
      {1, 2, 3},
      {4, 5},
      {6, 7, 8},
  };
  std::vector<int> phones2num_pdf_classes = {0, 3, 3, 3, 3,
                                             3, 3, 3, 3};  // size is 1 + 8
  ContextDependency* ctx_dep =
      MonophoneContextDependencyShared(shared_phones, phones2num_pdf_classes);
  ctx_dep->Write(std::cout, false);
  // clang-format off
  /*
   ContextDependency 1 0 ToPdf
   SE 0 [ 1 2 3 ]                               // key is 0, yes when phones are in [1, 2, 3]
    { TE -1 3 ( CE 0 CE 1 CE 2 ) }              // yes map
    SE 0 [ 4 5 ]                                // no map, key is 0, when phones are in [ 4, 5 ]
      { TE -1 3 ( CE 3 CE 4 CE 5 ) }            //   yes map
      TE -1 3 ( CE 6 CE 7 CE 8 )
   */
  // clang-format on

  Output o("tree2", false);
  ctx_dep->Write(o.Stream(), false);
  std::cout << "num pdfs: " << ctx_dep->NumPdfs() << "\n";  // num pdfs: 6
  o.Close();
  // draw-tree phones.txt tree | dot -Tpdf > tree.pdf
  // where phones.txt is
  // eps 0
  // A 1
  // B 2
}

int main() {
  // test();
  test2();
  return 0;
}
