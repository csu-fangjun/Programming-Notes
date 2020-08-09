// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

// Add 10 to the passed integer.
//
// @param a the first integer
// @return a + 10
// int inc(int a);
//
  .section .data

// int val = 10;
// In assembly, val is a label representing an address
val: .word 10

  .section .text
  .global inc
  .type inc, %function

inc:
  adr x1, val       // now x1 contains the address
  ldr w2, [x1]
  add w0, w0, w2
  ret
  .size inc, . - inc
