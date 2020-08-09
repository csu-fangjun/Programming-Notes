// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

// Compute the sum of a and b
//
// @param a the first integer
// @param b the second integer
// @return a  + b
// int add(int a, int b);
//
// Perhaps this is the simplest A64 assembly program.
//
// Parameter a is passed in the register `x0`;
// whereas parameter b is in register `x1`.
//
// The return value is in `x0`.

  .section .text
  .global add
  .type add, %function

add:
  add w0, w0, w1
  ret
  .size add, . - add
