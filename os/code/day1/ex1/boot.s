# on boot, it is in 16-bit mode

.code16

start:
  ljmp $0xffff, $0 # jumps to ffff:0000, i.e., the reset vector
                   # warm reboot

.fill 510 - (. - start), 1, 0
.word 0xaa55 # little endian, so it is 55 aa
