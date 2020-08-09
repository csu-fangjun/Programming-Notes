// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

  .section .data

/*
".equ", ".set", and "=" bind a const value to a symbol
*/

  .equ a, 1
  .set b, 2
c = 3

#define d 4


  .global test
  .type test, %function
// int test();
//  return a + b + c + d
test:
  mov w0, a
  add w0, w0, b
  add w0, w0, c
  add w0, w0, d
  ret
  .size test, . - test

