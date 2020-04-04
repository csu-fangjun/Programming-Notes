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
// 51: x0[31:16] = 0xdead
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
}

}  // namespace

void test_inline_asm() {
  test_move();
  test_add();
}
