# as --32 -o hello_world.o hello_world.s
# ld -m elf_i386 --dynamic-linker /lib/ld-linux.so.2 -o hello_world hello_world.o -lc

.data
s:
  .string "hello world"

.text
  .global _start
_start:
  push %ebp
  mov %esp, %ebp
  mov $s, %eax
  push %eax
  call puts
  xor %eax, %eax
  call exit
  ret
