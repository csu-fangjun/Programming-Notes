// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

#include <cassert>
#include <cstdint>

// aarch64-linux-gnu-g++ -std=c++11 -c inline_asm.cc
// aarch64-linux-gnu-objdump -d inline_asm.o

namespace {

void test_move() {
  {
    int a = 10;
    int b;
    asm volatile("mov %w0, %w1"
                 :        // output
                 "=r"(b)  // %0
                 :        // input
                 "r"(a)   // %1
                 :        // clobber list
                 "memory");
    assert(a == b);
// NOTE: %0 is 64-bit, whereas %w0 is 32-bit
//       since int is 32-bit, so we use %w0
//
//  8: move an immediate value 10 to w0
//  c: save w0 to memory; address of variable a is x29 + 44
//  10: load a to w0; this is the `input` of the inline asm;
//      the compiler has assigned register w0 to the `input`
//  14: this is the inline asm `mov %w0, %w0`; the compiler
//      also assigns register w0 to the output
//  15: this is the inline asm for output;
//      address of variable b is x29 + 40
//  1c: now it comes the `assert` statement; load a into w1
//  20: load b into w0
//  24: compare w1 and w0
#if 0
0:   a9bd7bfd        stp     x29, x30, [sp,#-48]!
4:   910003fd        mov     x29, sp
8:   52800140        mov     w0, #0xa                        // #10
c:   b9002fa0        str     w0, [x29,#44]
10:  b9402fa0        ldr     w0, [x29,#44]
14:  2a0003e0        mov     w0, w0
18:  b9002ba0        str     w0, [x29,#40]
1c:  b9402fa1        ldr     w1, [x29,#44]
20:  b9402ba0        ldr     w0, [x29,#40]
24:  6b00003f        cmp     w1, w0
28:  54000140        b.eq    50 <_ZN12_GLOBAL__N_19test_moveEv+0x50>
#endif
  }
  {
    int64_t a = 0x12345678deadbeef;
    int64_t b;
    asm volatile(
        "mov %0, %1     \n"
        "add %0, %0, 1  \n"
        : "=r"(b)  // %0
        : "r"(a)   // %1
        : "memory");
    assert((a + 1) == b);
// 50: x0[15:0] = 0xbeef
// 54: x0[31:16] = 0xdead
// 58: x0[47:32] = 0x5678
// 5c: x0[63:48] = 0x1234
// 60: save x0 to memory address x29 + 32 of variable a;
// 64: load a into x0; this is the inline asm for `input`;
//     the compiler has assigned x0 to the input variable
// 68: this is for inline asm `mov %0, %1`
//     the compiler has assigned x0 to the output variable
// 6c: this is for inline asm `add %0, %0, #1`
// 70: save x0 to the memory address x29 + 24 of variable b;
//     this is for the output variable of the inline asm
// 74: now it comes to assert ((a+1) == b)
//     load x29+32 to x0, i.e., load a into x0
// 78: x1 = x0 + 1 = a + 1
// 7c: load x29 + 24 into x0, i.e., load b into x0
// 80 compare x1 and x0, i.e., (a+1) and b
#if 0
50:   d297dde0        mov     x0, #0xbeef                     // #48879
54:   f2bbd5a0        movk    x0, #0xdead, lsl #16
58:   f2cacf00        movk    x0, #0x5678, lsl #32
5c:   f2e24680        movk    x0, #0x1234, lsl #48
60:   f90013a0        str     x0, [x29,#32]
64:   f94013a0        ldr     x0, [x29,#32]
68:   aa0003e0        mov     x0, x0
6c:   91000400        add     x0, x0, #0x1
70:   f9000fa0        str     x0, [x29,#24]
74:   f94013a0        ldr     x0, [x29,#32]
78:   91000401        add     x1, x0, #0x1
7c:   f9400fa0        ldr     x0, [x29,#24]
80:   eb00003f        cmp     x1, x0
84:   54000140        b.eq    ac <_ZN12_GLOBAL__N_19test_moveEv+0xac>
#endif
  }
}

void test_add() {
  {
    int a = 1;
    int b = 2;
    int c = 3;
    asm volatile(
        // add 1 to a
        "add %w0, %w0, 1    \n"
        // add 10 to b
        "add %w1, %w1, 10   \n"
        // add 100 to c
        "add %w2, %w2, 100  \n"
        :         // output
        "=r"(a),  // %0
        "=r"(b),  // %1
        "=r"(c)   // %2
        :         // input
        "0"(a),   // 0 means using output operand %0
        "1"(b),   // 1 means using output operand %1
        "2"(c)    // 2 means using output operand %2
        :         // clobber list
        "memory", "x0");
    assert(a == 2);
    assert(b == 12);
    assert(c == 103);
// bc: mov 1 to w0
// c0: save w0 to the memory address x29 + 28 of variable a
// c4: mov 2 to w0
// c8: save w0 to the memory address x29 + 24 of variable b
// cc: mov 3 to w0
// d0: save 20 to the memory address x29 + 20 of variable c
// d4: load memory x29 + 28, i.e., variable a into w2;
//     this is for inline asm; the compiler has assigned w2 to input 0
// d8: load memory x29 + 24, i.e., variable b into w1;
//     this is for inline asm; the compiler has assigned w1 to input 1
// dc: load memory x29 + 20, i.e., variable c into w0;
//     this is for inline asm; the compiler has assigned w0 to input 2
// e0: w3 = w2, the compiler has assigned w3 to output 0; this copies
//     the content from input 0 to output 0
// e4: w2 = w1, the compiler has assigned w2 to output 1; this copies
//     the content from input 1 to output 1
// e8: w1 = w0, the compiler has assigned w1 to output 3; this copies
//     the content from input 3 to output 3
//
// TODO(fangjun): why cannot input 0 and output 0 share the same register??
//
// ec: w3 = w3 + 1
// f0: w2 = w2 + 10
// f4: w1 = w1 + 100
//
// now save the result to the output variables
//
// f8:  save w3 to the memory address x29 + 28 of variable a
// fc:  save w2 to the memory address x29 + 24 of variable b
// 100: save w1 to the memory address x29 + 20 of variable c
//
#if 0
b4:   a9be7bfd        stp     x29, x30, [sp,#-32]!
b8:   910003fd        mov     x29, sp
bc:   52800020        mov     w0, #0x1                        // #1
c0:   b9001fa0        str     w0, [x29,#28]
c4:   52800040        mov     w0, #0x2                        // #2
c8:   b9001ba0        str     w0, [x29,#24]
cc:   52800060        mov     w0, #0x3                        // #3
d0:   b90017a0        str     w0, [x29,#20]
d4:   b9401fa2        ldr     w2, [x29,#28]
d8:   b9401ba1        ldr     w1, [x29,#24]
dc:   b94017a0        ldr     w0, [x29,#20]
e0:   2a0203e3        mov     w3, w2
e4:   2a0103e2        mov     w2, w1
e8:   2a0003e1        mov     w1, w0
ec:   11000463        add     w3, w3, #0x1
f0:   11002842        add     w2, w2, #0xa
f4:   11019021        add     w1, w1, #0x64
f8:   b9001fa3        str     w3, [x29,#28]
fc:   b9001ba2        str     w2, [x29,#24]
100:  b90017a1        str     w1, [x29,#20]
104:  b9401fa0        ldr     w0, [x29,#28]
108:  7100081f        cmp     w0, #0x2
#endif
  }
  {
    int a = 10;
    int b = 20;
    int sum = -1;
    asm volatile("add %[res], %[a], %[b]   \n"
                 // we can use an arbitrary symbol name instead of `res`
                 : [ res ] "=r"(sum)
                 : [ a ] "r"(a), [ b ] "r"(b)
                 : "memory");
    assert(sum == 30);
// 194: w0 = 10
// 198: save w0 to memory address x29 + 32 of variable a
// 19c: w0 = 20
// 1a0: save w0 to memory address x29 + 28 of variable b
// 1a4: w0 = -1
// 1a8: save w0 to memory address x29 + 24 of variable sum
// 1ac: w0 = *(x29 + 32) = a; the compiler has assigned w0
//      to the input variable a; note that the upper 32-bit
//      is set to zero automatically by `ldr`
// 1b0: w1 = *(x29 + 28) = b; the compiler has assigned w1
//      to the input variable b
// 1b4: x0 = x0 + x1 = a + b; the compiler has assigned w0
//      to the output variable sum. Note that it shares the
//      same register with the input variable a !
// 1b8: *(x29 + 24) = w0, save the result to variable sum
//
#if 0
194:   52800140        mov     w0, #0xa                        // #10
198:   b90023a0        str     w0, [x29,#32]
19c:   52800280        mov     w0, #0x14                       // #20
1a0:   b9001fa0        str     w0, [x29,#28]
1a4:   12800000        mov     w0, #0xffffffff                 // #-1
1a8:   b9001ba0        str     w0, [x29,#24]
1ac:   b94023a0        ldr     w0, [x29,#32]
1b0:   b9401fa1        ldr     w1, [x29,#28]
1b4:   8b010000        add     x0, x0, x1
1b8:   b9001ba0        str     w0, [x29,#24]
1bc:   b9401ba0        ldr     w0, [x29,#24]
1c0:   7100781f        cmp     w0, #0x1e
#endif
  }
}

void test_load_pair() {
  {
    int32_t a[4] = {10, 20, 30, 40};
    int32_t b[2] = {};
    int32_t c[2] = {};
    asm volatile(
        "ldp x0, x1, [%[a]]   \n"
        "str x0, [%[b]]       \n"
        "str x1, [%[c]]       \n"
        :
        : [ a ] "r"(a), [ b ] "r"(b), [ c ] "r"(c)
        : "memory", "x0", "x1");

    assert(b[0] == 10);
    assert(b[1] == 20);

    assert(c[0] == 30);
    assert(c[1] == 40);

// ldp x0, x1, [x2]
//    load the first 8 bytes to x0
//    load the **next** 8 bytes to x1
//
// 1fc: w0 = 10
// 200: a[0] = w0 = 10
// 204: w0 = 20
// 208: a[1] = w0 = 20
// 20c: w0 = 30
// 210: a[2] = w0 = 30
// 214: w0 = 40
// 218: a[3] = w0 = 40
// 21c: *(x29 + 24) = 0, set b[0] and b[1] to 0
// 220: *(x29 + 16) = 0, set c[0] and c[1] to 0
// 224: x2 = x29 + 32, the address of a
// 228: x3 = x29 + 24, the address of b
// 22c: x4 = x29 + 16, the address of c
// 230: load a[0], a[1] to x0; load a[2], a[3] to x1
// 234: save x0, i.e., a[0], a[1], to b[0], b[1]
// 238: save x1, i.e., a[2], a[3], to c[0], c[1]
#if 0
1fc:   52800140        mov     w0, #0xa                        // #10
200:   b90023a0        str     w0, [x29,#32]
204:   52800280        mov     w0, #0x14                       // #20
208:   b90027a0        str     w0, [x29,#36]
20c:   528003c0        mov     w0, #0x1e                       // #30
210:   b9002ba0        str     w0, [x29,#40]
214:   52800500        mov     w0, #0x28                       // #40
218:   b9002fa0        str     w0, [x29,#44]
21c:   f9000fbf        str     xzr, [x29,#24]
220:   f9000bbf        str     xzr, [x29,#16]
224:   910083a2        add     x2, x29, #0x20
228:   910063a3        add     x3, x29, #0x18
22c:   910043a4        add     x4, x29, #0x10
230:   a9400440        ldp     x0, x1, [x2]
234:   f9000060        str     x0, [x3]
238:   f9000081        str     x1, [x4]
23c:   b9401ba0        ldr     w0, [x29,#24]
240:   7100281f        cmp     w0, #0xa
#endif
  }
}

// dup Vd.<Td> Vn.Ts[index]
// <Td>/<Ts> may be:
//   8B/B, 16B/B, 4H/H, 2S/S, 4S/S, 2D/D
//   index is in the range 0 to <Ts> - 1
//
// Duplicate single lane from Vn to every lane in Vd
void test_dup() {
  {
    int32_t a[4] = {10, 20, 30, 50};
    int32_t b[1] = {8};
    int32_t sum[4];
    asm volatile(
        // "ldr q0, [%[a]]               \n"
        "ld1 {v0.4s}, [%[a]]       \n"
        "ldr s1, [%[b]]            \n"
        "dup v2.4s, v1.s[0]        \n"
        "add v1.4s, v0.4s, v2.4s   \n"
        "st1 {v1.4s}, [%[sum]]     \n"
        // "str q1, [%[sum]]     \n"
        :  // no output
        : [ a ] "r"(a), [ b ] "r"(b), [ sum ] "r"(sum)
        : "memory", "v0", "v1", "v2");
    assert(sum[0] == 18);
    assert(sum[1] == 28);
    assert(sum[2] == 38);
    assert(sum[3] == 58);
//  Note: we can use
//+ either
//    ldr q0, [%[a]]
//+ or
//    ld1 {v0.4s}, [%[a]]
//
//  Similarly, we can use
//+ either
//    str q1, [%[sum]]
//+ or
//    st1 {v1.4s}, [%[sum]]
//
//  Notation:
//    b0 (1-byte), h0 (2-byte), s0 (4-byte), d0 (8-byte), q0 (16-byte)
//    all of them refer to the same register v0
//
//  334: it uses x0 = x29 + 0x30 to get the address of a[0]
//  340: we can use `ld1 {v0.4s}, [x0]` to load
//+      4 32-bit from memory into v0.
//  344: Note that we can use `ldr s1, [x1]` to load a value
//+      from memory into s1.
//  348: v2.4s occupies only 8-byte, the upper 8-byte is not used.
// +     It sets v2.4s[0], v2.4s[1], v2.4s[2] and v2.4s[3] to v1.s[0].
#if 0
304:   a9bc7bfd        stp     x29, x30, [sp,#-64]!
308:   910003fd        mov     x29, sp
30c:   52800140        mov     w0, #0xa                        // #10
310:   b90033a0        str     w0, [x29,#48]
314:   52800280        mov     w0, #0x14                       // #20
318:   b90037a0        str     w0, [x29,#52]
31c:   528003c0        mov     w0, #0x1e                       // #30
320:   b9003ba0        str     w0, [x29,#56]
324:   52800640        mov     w0, #0x32                       // #50
328:   b9003fa0        str     w0, [x29,#60]
32c:   52800100        mov     w0, #0x8                        // #8
330:   b9002ba0        str     w0, [x29,#40]
334:   9100c3a0        add     x0, x29, #0x30
338:   9100a3a1        add     x1, x29, #0x28
33c:   910063a2        add     x2, x29, #0x18
340:   4c407800        ld1     {v0.4s}, [x0]
344:   bd400021        ldr     s1, [x1]
348:   4e040422        dup     v2.4s, v1.s[0]
34c:   4ea28401        add     v1.4s, v0.4s, v2.4s
350:   4c007841        st1     {v1.4s}, [x2]
354:   b9401ba0        ldr     w0, [x29,#24]
358:   7100481f        cmp     w0, #0x12
#endif
  }
  {
    int16_t a[4] = {1, 2, 3, 5};
    int8_t b = 10;
    int16_t c[4];
    asm volatile(
        // "ldr d0, [%[a]]        \n"
        "ld1 {v0.4h}, [%[a]]      \n"
        "dup v2.4h, %w[b]         \n"
        "add v1.4h, v0.4h, v2.4h  \n"
        "st1 {v1.4h}, [%[c]]      \n"
        :  // no output
        : [ a ] "r"(a), [ b ] "r"(b), [ c ] "r"(c)
        : "memory", "v0", "v1", "v2");
    assert(c[0] == 11);
    assert(c[1] == 12);
    assert(c[2] == 13);
    assert(c[3] == 15);
//
//  dup Vd.<T>, Wn
//+  <T> may be 8B, 16B, 4H, 8H, 2S or 4S;
//+  It uses lower bits of Wn for replication;
//+  higher bits are dropped.
//
//  414: w0 = 1; Note that even though we use int16_t in c,
//+      the assembler still uses 32-bit w0.
//  418: It uses `strh` to save the lower 16-bit of w0
//+      to the memory address x29 + 24 of a[0]
#if 0
410:   94000000        bl      0 <__assert_fail>
414:   52800020        mov     w0, #0x1                        // #1
418:   790033a0        strh    w0, [x29,#24]
41c:   52800040        mov     w0, #0x2                        // #2
420:   790037a0        strh    w0, [x29,#26]
424:   52800060        mov     w0, #0x3                        // #3
428:   79003ba0        strh    w0, [x29,#28]
42c:   528000a0        mov     w0, #0x5                        // #5
430:   79003fa0        strh    w0, [x29,#30]
434:   52800140        mov     w0, #0xa                        // #10
438:   b9004fa0        str     w0, [x29,#76]
43c:   910063a0        add     x0, x29, #0x18
440:   b9404fa1        ldr     w1, [x29,#76]
444:   910043a2        add     x2, x29, #0x10
448:   0c407400        ld1     {v0.4h}, [x0]
44c:   0e020c22        dup     v2.4h, w1
450:   0e628401        add     v1.4h, v0.4h, v2.4h
454:   0c007441        st1     {v1.4h}, [x2]
458:   79c023a0        ldrsh   w0, [x29,#16]
45c:   71002c1f        cmp     w0, #0xb
#endif
  }
}

void test_swap_endianess() {
  int32_t a = 0x12345678;
  asm volatile("rev %w[dst], %w[src] \n" : [ dst ] "=r"(a) : [ src ] "0"(a) :);
  assert(a == 0x78563412);

  a = 0x1234;
  asm volatile("rev %w[dst], %w[src] \n" : [ dst ] "=r"(a) : [ src ] "0"(a) :);
  assert(a == 0x34120000);

  int16_t b = 0x1234;
  asm volatile("rev16 %w[dst], %w[src] \n"
               : [ dst ] "=r"(b)
               : [ src ] "0"(b)
               :);
  assert(b == 0x3412);

  b = 0x1200;
  asm volatile("rev16 %w[dst], %w[src] \n"
               : [ dst ] "=r"(b)
               : [ src ] "0"(b)
               :);
  assert(b == 0x12);
}

// bit clear
void test_bic() {
  int32_t a = 1 | (1 << 2) | (1 << 4) | (1 << 8);
  int32_t mask = 1 << 4;
  int32_t b;

  // dst = src & (~mask)
  asm volatile("bic %w[dst], %w[src], %w[mask] \n"
               : [ dst ] "=r"(b)
               : [ src ] "r"(a), [ mask ] "r"(mask)
               :);
  assert(b == (1 | (1 << 2) | (1 << 8)));

  a = 10;
  mask = 0x03;
  asm volatile("bic %w[dst], %w[src], %w[mask] \n"
               : [ dst ] "=r"(b)
               : [ src ] "r"(a), [ mask ] "r"(mask)
               :);
  assert(b == 8);
}

// logical or
void test_orr() {
  int32_t a = 0x01;
  int32_t b = 0x03;
  int32_t c;
  asm volatile("orr %w[dst], %w[src1], %w[src2]   \n"
               : [ dst ] "=r"(c)
               : [ src1 ] "r"(a), [ src2 ] "r"(b)
               :);
  assert(c == (0x01 | 0x03));
}

// eor: exclusive or
void test_eor() {
  int32_t a = 0x01;
  int32_t b = 0x03;
  int32_t c;
  asm volatile("eor %w[dst], %w[src1], %w[src2]   \n"
               : [ dst ] "=r"(c)
               : [ src1 ] "r"(a), [ src2 ] "r"(b)
               :);
  assert(c == (0x01 ^ 0x03));
}

// res = src1 | (~src2)
void test_orn() {
  int32_t a = 0x01;
  int32_t b = 0x03;
  int32_t c;
  asm volatile("orn %w[dst], %w[src1], %w[src2]   \n"
               : [ dst ] "=r"(c)
               : [ src1 ] "r"(a), [ src2 ] "r"(b)
               :);
  assert(c == (0x01 | ~0x03));
}

// cmp x0, x1, 0, ge
//  if previous condition is ge; then
//    flag = cmp(x0, x1)
//  else
//    flag = 0  // because we provide 0 here
void test_ccmp() {
  int a = 2;
  int b = 1;
  int c = 1;
  int d;

  if ((a > b) && (b == c)) {
    d = 10;
  } else {
    d = 100;
  }

  int f = -1;
  // cmp %w[a], %w[b]
  // if gt; then
  //   flag = cmp(%w[b], %w[c])
  // else
  //   flag = 0
  //
  // In our case, b > c, so Z is 1, b.eq is taken
  asm volatile(
      "cmp %w[a], %w[b]         \n"
      "ccmp %w[b], %w[c], 0, gt \n"
      "b.eq 1f                  \n"
      "mov %w[f], 100           \n"
      "b 2f                     \n"
      "1:                       \n"
      "mov %w[f], 10            \n"
      "2:                       \n"
      : [ f ] "=r"(f)
      : [ a ] "r"(a), [ b ] "r"(b), [ c ] "r"(c)
      :);
  assert(f == d);
}

// count leading zero
void test_clz() {
  int32_t a = 0x01;
  int r;

  asm volatile("clz %w[dst], %w[src]\n" : [ dst ] "=r"(r) : [ src ] "r"(a) :);

  assert(r == 31);
}

// csel x0, true, false, cond
void test_csel() {
  int32_t a = 10;
  int32_t b = 20;
  int32_t r;

  r = (a >= 10) ? a : b;

  int f;

  asm volatile(
      "cmp %w[a], 10                \n"
      "csel %w[f], %w[a], %w[b], ge \n"
      : [ f ] "=r"(f)
      : [ a ] "r"(a), [ b ] "r"(b)
      :);
  assert(r == f);
}

}  // namespace

void test_inline_asm() {
  test_move();
  test_add();
  test_load_pair();
  test_dup();
  test_swap_endianess();
  test_bic();
  test_orr();
  test_eor();
  test_orn();
  test_ccmp();
  test_clz();
  test_csel();
}
