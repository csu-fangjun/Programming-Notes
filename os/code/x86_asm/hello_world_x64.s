# gcc -o hello_world hello_world.s

  .data
.LC0:
  .string "hello word"

  .text
  .global _main
_main:
  push %rbp
  mov %rsp, %rbp
  lea .LC0(%rip), %rdi
  call _puts
  leave
  ret

