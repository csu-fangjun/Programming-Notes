
.code32
.section .text
  .global main
main:
  mov $0x01, %eax
	call f
  int $0x80

.section .text.foo
	.global f
f:
  mov $0x02, %ebx
	ret
