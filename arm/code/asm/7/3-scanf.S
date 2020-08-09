/*
The program use scanf to accept two intergers
from the terminal and print their sum.

*/
  .section .data
input_msg1:
  .string "please enter a: "

input_msg2:
  .string "please enter b: "

format_str:
  .string "%d"

result_str:
  .string "%d + %d = %d\n"

  .section .data
  .align 2    // for arm, 2 means the lower 2-bit is 0
a: .word 0

  .balign 4   // byte align, so b is 4-byte aligned
b: .word 0

  .section .text
  .global main
  .type main, %function

main:
  stp x29, x30, [sp, -16]!

  adr x0, input_msg1
  bl printf

  adr x0, format_str
  adr x1, a
  bl scanf

  adr x0, input_msg2
  bl printf

  adr x0, format_str
  adr x1, b
  bl scanf

  adr x0, a
  ldr x1, [x0]

  adr x0, b
  ldr x2, [x0]

  add x3, x1, x2
  adr x0, result_str
  bl printf

  mov x0, 0
  ldp x29, x30, [sp], 16
  ret

