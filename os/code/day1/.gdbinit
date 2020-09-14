set architecture i8086
target remote localhost:1234
layout regs
b *0x7c00
b *0x7c24
b *0x600
