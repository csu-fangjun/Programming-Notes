
.code16
# the compiled object is saved in head 0, track 0, sector 1
# it will be loaded to 0x0050:0000, e.g., 0x500

	mov $0x1234, %ax
	xor %ax, %ax
	inc %ax

  # http://www.ablmcc.edu.hk/~scy/CIT/8086_bios_and_dos_interrupts.htm#int10h_0Eh
  mov $0x0e, %ah # write a charater in al to the console

  lea msg, %si
  mov $msg_len, %cx # number of characters, loop counter

1:
	lodsb
	int $0x10
	loop 1b
	hlt

msg:
	.string "hello my OS"
	.equ msg_len, . - msg

