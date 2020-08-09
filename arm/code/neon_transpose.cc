// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)
//
//  References:
//   https://github.com/Maratyszcza/NNPACK/blob/master/src/neon/transpose.h
#include <arm_neon.h>

#include <cassert>
#include <iostream>
#include <vector>
namespace {
/**
 * Transpose a 4x4 matrix in-place.
 *
 * This is the reference implementation.
 * @param a a 4x4 matrix
 */
void tranpose4x4_ref(float* a) {
  for (int r = 0; r != 4; ++r) {
    for (int c = r + 1; c != 4; ++c) {
      std::swap(a[r * 4 + c], a[c * 4 + r]);
    }
  }
}

void tranpose4x4_neon(float* a) {
  // [row00, row01, row02, row03]
  float32x4_t row0 = vld1q_f32(a);

  // [row10, row11, row12, row13]
  float32x4_t row1 = vld1q_f32(a + 4);

  // [row20, row21, row22, row23]
  float32x4_t row2 = vld1q_f32(a + 8);

  // [row30, row31, row32, row33]
  float32x4_t row3 = vld1q_f32(a + 12);

  // trn is short for transpose
  //
  // place the elements from row0 in even numbered position
  // place the elements from row1 in odd numbered position
  // r01 is [row00, row10, row02, row12]
  //        [row01, row11, row03, row13]
  float32x4x2_t r01 = vtrnq_f32(row0, row1);

  // r23 is [row20, row30, row22, row32]
  //        [row21, row31, row23, row33]
  float32x4x2_t r23 = vtrnq_f32(row2, row3);

  row0 = vcombine_f32(vget_low_f32(r01.val[0]), vget_low_f32(r23.val[0]));
  row1 = vcombine_f32(vget_low_f32(r01.val[1]), vget_low_f32(r23.val[1]));
  row2 = vcombine_f32(vget_high_f32(r01.val[0]), vget_high_f32(r23.val[0]));
  row3 = vcombine_f32(vget_high_f32(r01.val[1]), vget_high_f32(r23.val[1]));
  vst1q_f32(a, row0);
  vst1q_f32(a + 4, row1);
  vst1q_f32(a + 8, row2);
  vst1q_f32(a + 12, row3);
}

}  // namespace

void test_transpose() {
  std::vector<float> a = {
      // clang-format off
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
      // clang-format on
  };

  std::vector<float> b(a);

  tranpose4x4_ref(a.data());
  tranpose4x4_neon(b.data());

  for (int i = 0; i != 16; ++i) {
    assert(a[i] == b[i]);
  }

#if 0
  for (int i = 0; i != 16; ++i) {
    std::cout << b[i] << " ";
    if ((i + 1) % 4 == 0) {
      std::cout << "\n";
    }
  }
// 0 4 8 12
// 1 5 9 13
// 2 6 10 14
// 3 7 11 15
#endif
}
