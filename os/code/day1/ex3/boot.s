# print a string: hello world

.code16

start:
  xor %ax, %ax
  mov %ax, %ss
  mov %ax, %es
  mov %ax, %ds

  mov $msg, %si
  mov $len, %cx # number of characters, loop counter

  # http://www.ablmcc.edu.hk/~scy/CIT/8086_bios_and_dos_interrupts.htm#int10h_0Eh
  mov $0x0e, %ah # write a charater in al to the console

1:
  lodsb
  int $0x10
  loop 1b

  hlt

msg:
  .string "hello world\n"

  .equ len, . - msg

.fill 510 - (. - start), 1, 0
.word 0xaa55 # little endian, so it is 55 aa
