// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

// Add 10 to the passed integer.
//
// @param a the first integer
// @return a + val
// @note val is a global variable
// int inc(int a);
//
  .section .data

  .align 2 // 2 means pow(2, 2)
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

  // int get_val();
  .global get_val
  .type get_val, %function

get_val:
  adr x1, val
  ldr w0, [x1]
  ret
  .size get_val, . - get_val

  // void set_val(int v);
  .global set_val
  .type set_val, %function
set_val:
  adr x1, val
  str w0, [x1]
  ret
  .size set_val, . -  set_val


