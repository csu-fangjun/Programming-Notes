
# for IA32, the system call is invoked via `int 0x80`
#
# eax: the system call number
# syscall(ebx,  ecx,  edx,  esi,  edi,  ebp)
#         arg1, arg2  arg3  arg4  arg5  arg6

.code32
	.global main
main:
	mov $0x01, %eax
	mov $0x03, %ebx #  exit(0x03);
	int $0x80
