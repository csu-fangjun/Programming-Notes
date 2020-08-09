#include <arm_neon.h>

#include <cassert>
#include <iostream>

namespace {

// https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics?search=vld1q_s8
// int8x16_t vld1q_s8 (int8_t const * ptr)
//
//  LD1 {Vt.16B},[Xn]
//
//  The disassembler uses
//+    ldr q0, [x0]
//
//  It is equivalent to
//+    ld1 {v0.16b}, [x0]
//
void test_vld1q_s8() {
  int8_t a[16] = {12};
  int8x16_t b = vld1q_s8(a);

// 0: allocate 48 bytes
// 4: load the address of a into x0
// 8: set all elements of a to 0
// c: w0 = 12
// 10: a[0] = w0 = 12
// 14: x0 = sp + 8 = &a[0]
// 18: *(sp + 24) = x0
// 1c: x0 = *(sp + 24)    (fangjun): why are 18 and 1c is here??
// 20: load 16-byte to q0 from address stored in x0
//     this single instruction load 16-byte into q0.
//     Note that q0 is the same register as v0.
// 24: save qo to (sp+32), which is the address of variable b
// 28: restore sp
// 2c: return
#if 0
0:   d100c3ff        sub     sp, sp, #0x30
4:   910023e0        add     x0, sp, #0x8
8:   a9007c1f        stp     xzr, xzr, [x0]
c:   52800180        mov     w0, #0xc                        // #12
10:  390023e0        strb    w0, [sp,#8]
14:  910023e0        add     x0, sp, #0x8
18:  f9000fe0        str     x0, [sp,#24]
1c:  f9400fe0        ldr     x0, [sp,#24]
20:  3dc00000        ldr     q0, [x0]
24:  3d800be0        str     q0, [sp,#32]
28:  9100c3ff        add     sp, sp, #0x30
2c:  d65f03c0        ret
#endif
}
void test_vld1q_lane_s32() {
  int32_t a = 10;
  int32x4_t b = {0, 1, 2, 3};
  // First, copies b to the result
  // Second, load the value from a into the specified lane
  int32x4_t c = vld1q_lane_s32(&a, b, 0);
  assert(c[0] == 10);
  assert(c[1] == 1);
  assert(c[2] == 2);
  assert(c[3] == 3);

  c = vld1q_lane_s32(&a, b, 3);
  assert(c[0] == 0);  // c[0] is changed!
  assert(c[1] == 1);
  assert(c[2] == 2);
  assert(c[3] == 10);
}

void test_vaddq_s32() {
  int32x4_t a = {10, 20, 30, 50};
  int32x4_t b = {1, 2, 3, 5};
  int32x4_t c = vaddq_s32(a, b);
  assert(c[0] == 11);
  assert(c[1] == 22);
  assert(c[2] == 33);
  assert(c[3] == 55);
}

void test_reciprocal() {
  float32x4_t a = {1, 2, 3, 5};
  float32x4_t b = vrecpeq_f32(a);
  // printf("%.6f, %.6f, %.6f, %.6f\n", b[0], b[1], b[2], b[3]);
  // 0.998047, 0.499023, 0.333008, 0.199707

  float32x4_t c = vrecpsq_f32(a, b);
  b = vmulq_f32(c, b);
  // printf("%.6f, %.6f, %.6f, %.6f\n", b[0], b[1], b[2], b[3]);
  // 0.999996, 0.499998, 0.333333, 0.200000

  c = vrecpsq_f32(a, b);
  b = vmulq_f32(c, b);
  // printf("%.6f, %.6f, %.6f, %.6f\n", b[0], b[1], b[2], b[3]);
  // 1.000000, 0.500000, 0.333333, 0.200000

  // 2-1*1=1, 2-2*2=-2, 2-3*3=-7, 2-5*5=-23
  c = vrecpsq_f32(a, a);
  // printf("%.6f, %.6f, %.6f, %.6f\n", c[0], c[1], c[2], c[3]);
  // 1.000000, -2.000000, -7.000000, -23.000000
}

void test_leaky_relu_32x2_t() {
  // this is modified from
  // https://github.com/Maratyszcza/NNPACK/blob/bda381b2e207230cab2b38be310a17831cfe384a/include/nnpack/activations.h#L37
  float32x2_t a = {-1, 2};
  float32_t slope = 0.125;

  // bit pattern is kept, no rounding here
  int32x2_t b = vreinterpret_s32_f32(a);

  int32x2_t sign = vshr_n_s32(b, 31);
  // we have to convert it to uint32x2_t
  // because the following "bitwise select" requires that.
  uint32x2_t negative_mask = vreinterpret_u32_s32(sign);

  float32x2_t activation = vmul_n_f32(a, slope);
  float32x2_t out = vbsl_f32(negative_mask, activation, a);
  // printf("%.6f, %.6f\n", out[0], out[1]);
  // -0.125000, 2.000000
}

void test_leaky_relu_32x4_t() {
  float32x4_t a = {-1, 2, -3, 5};
  float32_t slope = 0.125;

  int32x4_t b = vreinterpretq_s32_f32(a);
  int32x4_t sign = vshrq_n_s32(b, 31);
  uint32x4_t negative_mask = vreinterpretq_u32_s32(sign);
  float32x4_t activation = vmulq_n_f32(a, slope);
  float32x4_t out = vbslq_f32(negative_mask, activation, a);
  // printf("%.6f, %.6f, %.6f, %.6f\n", out[0], out[1], out[2], out[3]);
  // -0.125000, 2.000000, -0.375000, 5.000000
}

void test_vmlaq_f32() {
  float32x4_t a = {1, 2, 3, 5};
  float32x4_t b = {10, 20, 30, 50};
  float32x4_t c = {6, 7, 8, 9};
  // multiply-accumulate
  float32x4_t d = vmlaq_f32(a, b, c);  // a + b * c
  // printf("%.6f, %.6f, %.6f, %.6f\n", d[0], d[1], d[2], d[3]);
  // 61.000000, 142.000000, 243.000000, 455.000000

  //  FMLA Vd.4S,Vn.4S,Vm.4S
  float32x4_t e = vfmaq_f32(a, b, c);
  // printf("%.6f, %.6f, %.6f, %.6f\n", e[0], e[1], e[2], e[3]);
  //
  // NOTE:
  //   - vmlaq_f32 is implemented as: a + b * c, which requires more than 1
  //+    instruction
  //   - vfmaq_f32 is implemented as one instruciton: `fmla`
}

}  // namespace

void test_neon_intrinsics() {
  test_vld1q_s8();
  test_vld1q_lane_s32();
  test_vaddq_s32();
  test_reciprocal();
  test_leaky_relu_32x2_t();
  test_leaky_relu_32x4_t();
  test_vmlaq_f32();

  void test_dot_product();
  test_dot_product();
  return;
}
