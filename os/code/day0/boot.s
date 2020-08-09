.code16

start:
  mov $0x0e, %ah # display character
  mov $0x30, %al # display character 0x30
  int $0x10

  mov $0x0e, %ah # display character
  mov $0x31, %al # display character 0x30
  int $0x10

  hlt

.fill 510 - (. - start), 1, 0
.word 0xaa55 # little endian, so it is 55 aa
