	.file	"datatype.c"
	.text
	.globl	c
	.data
	.type	c, @object
	.size	c, 1
c:
	.byte	1
	.globl	s
	.align 2
	.type	s, @object
	.size	s, 2
s:
	.value	2
	.globl	i
	.align 4
	.type	i, @object
	.size	i, 4
i:
	.long	4
	.globl	ia
	.align 16
	.type	ia, @object
	.size	ia, 20
ia:
	.long	1
	.long	2
	.long	3
	.zero	8
	.globl	str
	.section	.rodata
.LC0:
	.string	"hello"
	.section	.data.rel.local,"aw",@progbits
	.align 8
	.type	str, @object
	.size	str, 8
str:
	.quad	.LC0
	.section	.rodata
	.type	str_a, @object
	.size	str_a, 6
str_a:
	.string	"hello"
	.text
	.globl	add
	.type	add, @function
add:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %edx
	movl	-8(%rbp), %eax
	addl	%edx, %eax
	popq	%rbp
	ret
	.size	add, .-add
	.ident	"GCC: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0"
	.section	.note.GNU-stack,"",@progbits
