// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

  .section .data

hello:
  .asciz "hello"

world:
  .ascii "world"
  .byte 0

  .section .text
  .global say_hello
  .type say_hello, %function

say_hello:
  adr x0, hello
  ret
  .size say_hello, . - say_hello

  .global say_world
  .type say_world, %function
say_world:
  adr x0, world
  ret
  .size say_world, . - say_world

