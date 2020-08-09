// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)
//
//  References:
//   https://github.com/Maratyszcza/NNPACK/blob/master/src/neon/blas/sdotxf.c

#include <arm_neon.h>

#include <cassert>

namespace {
/**
 * Compute dot product between `a` and `b`.
 *
 * This is the reference implementation.
 *
 * @param n length of the vector
 * @param a a vector with `n` elements
 * @param b a vector with `n` elements
 *
 * @return the dot product of a and b
 */
float dot_product_ref(int n, const float* a, const float* b) {
  float sum = 0;
  for (int i = 0; i != n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Compute dot product between `a` and `b`.
 *
 * This is the NEON intrinsics implementation.
 *
 * @param n length of the vector
 * @param a a vector with `n` elements
 * @param b a vector with `n` elements
 *
 * @return the dot product of a and b
 */
float dot_product_neon(int n, const float* a, const float* b) {
  float sum = 0;
  float32x4_t accq = vdupq_n_f32(0);
  // float32x4_t accq = vmovq_n_f32(0);  // is also fine.
  for (; n >= 4; n -= 4) {
    const float32x4_t x = vld1q_f32(a);
    a += 4;

    const float32x4_t y = vld1q_f32(b);
    b += 4;

    accq = vfmaq_f32(accq, x, y);
  }

  float32x2_t acc = vadd_f32(vget_low_f32(accq), vget_high_f32(accq));
  if (n >= 2) {
    n -= 2;
    float32x2_t x = vld1_f32(a);
    a += 2;

    float32x2_t y = vld1_f32(b);
    b += 2;

    acc = vfma_f32(acc, x, y);
  }
  acc = vpadd_f32(acc, acc);
  // lane 0 and lane 1 are the same, we need only lane0
  //  vpadd_f32 is preferred over
  //    vget_lane_f32(acc, 0) + vget_lane_f32(acc, 1)

  if (n) {
    float32x2_t x = vld1_dup_f32(a);
    float32x2_t y = vld1_dup_f32(b);
    acc = vfma_f32(acc, x, y);
  }
  vst1_lane_f32(&sum, acc, 0);
  // vst1_lane_f32(&sum, acc, 1);  // is also fine
  return sum;
}

}  // namespace

void test_dot_product() {
  {
    constexpr int kN = 10;
    float a[kN];
    float b[kN];
    for (int i = 0; i != kN; ++i) {
      a[i] = i;
      b[i] = 3 * i + 5;
    }

    float sum1 = dot_product_ref(kN, a, b);
    float sum2 = dot_product_neon(kN, a, b);
    assert(sum1 == sum2);
  }
}
