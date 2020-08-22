# print a string: hello world

.code16

start:
# http://www.ablmcc.edu.hk/~scy/CIT/8086_bios_and_dos_interrupts.htm#int10h_0Eh
# al: character to write
  mov $0x0e, %ah  # teletype output
  mov $'h', %al
  int $0x10

  mov $'e', %al
  int $0x10

  mov $'l', %al
  int $0x10

  mov $'l', %al
  int $0x10

  mov $'o', %al
  int $0x10

  hlt


.fill 510 - (. - start), 1, 0
.word 0xaa55 # little endian, so it is 55 aa
