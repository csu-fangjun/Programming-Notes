
.code16

start:
  cli
  cld

  mov $hint, %si
  mov $hint_len, %cx # number of characters, loop counter

  # http://www.ablmcc.edu.hk/~scy/CIT/8086_bios_and_dos_interrupts.htm#int10h_0Eh
  mov $0x0e, %ah # write a charater in al to the console

1:
  lodsb
  int $0x10
  loop 1b

  mov $0x50, %ax
  mov %ax, %es
  xor %bx, %bx

  # the content is read into es:bx, e.g., 0x500

  # http://www.ablmcc.edu.hk/~scy/CIT/8086_bios_and_dos_interrupts.htm#int13h_02h
  mov $1, %al # number of sectors to read
  mov $0, %ch # cylinder number, i.e., track number
  mov $2, %cl # sector number, count from 1!
  mov $0, %dh # head number
  mov $0, %dl # drive number

  mov $0x02, %ah
  int $0x13

  ljmp $0x00, $0x500 # ljmp %cs, %ip

  # go to sample.s


hint:
  # note that sector number counts from 1.
  # the boot sector is sector 1
  .string "loading OS from head 0, track 0, sector 2\r\n"

  .equ hint_len, . - hint

.fill 510 - (. - start), 1, 0
.word 0xaa55 # little endian, so it is 55 aa
