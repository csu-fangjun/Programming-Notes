# gcc -m32 -o hello_main hello_main.s

.data
s:
  .string "hello world"

.text
  .global main
main:
  push %ebp
  mov %esp, %ebp
  mov $s, %eax
  push %eax
  call puts
  xor %eax, %eax
  leave
  ret
