
# as -o boot boot.s
# objdump -m i8086 -d boot
# .code16

abc:
  .long 1234
def:
  .long 5678

#.altmacro
.macro  sum from
  mov \from, %rax
  #mov from, %rax
.endm

sum $0xbeaf
sum $abc
#sum def
#mov g, %rbx

.globl _start
_start:

f:
  xor %ax, %ax
  xor %ax, %ax
  mov $'9', %ax
g:
  xor %ax, %ax
  xor %ax, %ax

