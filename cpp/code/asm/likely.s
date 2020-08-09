	.file	"likely.c"
	.text
	.globl	test_likely
	.type	test_likely, @function
test_likely:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	%edi, -4(%rbp)
	movl	-4(%rbp), %eax
	addl	$1, %eax
	popq	%rbp
	ret
	.size	test_likely, .-test_likely
	.globl	test_unlikely
	.type	test_unlikely, @function
test_unlikely:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	%edi, -4(%rbp)
	movl	-4(%rbp), %eax
	addl	$2, %eax
	popq	%rbp
	ret
	.size	test_unlikely, .-test_unlikely
	.globl	test0
	.type	test0, @function
test0:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$8, %rsp
	movl	%edi, -4(%rbp)
	cmpl	$0, -4(%rbp)
	je	.L6
	movl	-4(%rbp), %eax
	movl	%eax, %edi
	call	test_likely
	jmp	.L7
.L6:
	movl	-4(%rbp), %eax
	movl	%eax, %edi
	call	test_unlikely
.L7:
	leave
	ret
	.size	test0, .-test0
	.globl	test1
	.type	test1, @function
test1:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$8, %rsp
	movl	%edi, -4(%rbp)
	cmpl	$0, -4(%rbp)
	setne	%al
	movzbl	%al, %eax
	testq	%rax, %rax
	je	.L9
	movl	-4(%rbp), %eax
	movl	%eax, %edi
	call	test_likely
	jmp	.L10
.L9:
	movl	-4(%rbp), %eax
	movl	%eax, %edi
	call	test_unlikely
.L10:
	leave
	ret
	.size	test1, .-test1
	.globl	test2
	.type	test2, @function
test2:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$8, %rsp
	movl	%edi, -4(%rbp)
	cmpl	$0, -4(%rbp)
	setne	%al
	movzbl	%al, %eax
	testq	%rax, %rax
	je	.L12
	movl	-4(%rbp), %eax
	movl	%eax, %edi
	call	test_unlikely
	jmp	.L13
.L12:
	movl	-4(%rbp), %eax
	movl	%eax, %edi
	call	test_likely
.L13:
	leave
	ret
	.size	test2, .-test2
	.ident	"GCC: (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609"
	.section	.note.GNU-stack,"",@progbits
