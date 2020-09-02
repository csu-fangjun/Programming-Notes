	.file	"atexit-test.c"
	.section	.rodata
.LC0:
	.string	"in f1"
	.text
	.globl	f1
	.type	f1, @function
f1:
.LFB2:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$.LC0, %edi
	call	puts
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	f1, .-f1
	.section	.rodata
.LC1:
	.string	"in f2"
	.text
	.globl	f2
	.type	f2, @function
f2:
.LFB3:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$.LC1, %edi
	call	puts
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	f2, .-f2
	.section	.rodata
.LC2:
	.string	"in b2"
	.text
	.globl	b2
	.type	b2, @function
b2:
.LFB4:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$.LC2, %edi
	call	puts
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	b2, .-b2
	.section	.init_array,"aw"
	.align 8
	.quad	b2
	.section	.rodata
.LC3:
	.string	"in b1"
	.text
	.globl	b1
	.type	b1, @function
b1:
.LFB5:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$.LC3, %edi
	call	puts
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	b1, .-b1
	.section	.init_array
	.align 8
	.quad	b1
	.section	.rodata
.LC4:
	.string	"in b3 ctors"
	.text
	.globl	b3
	.type	b3, @function
b3:
.LFB6:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$.LC4, %edi
	call	puts
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	b3, .-b3
	.globl	b4
	.section	.ctors,"aw",@progbits
	.align 8
	.type	b4, @object
	.size	b4, 8
b4:
	.quad	b3
	.section	.rodata
.LC5:
	.string	"in b5 dtors"
	.text
	.globl	b5
	.type	b5, @function
b5:
.LFB7:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$.LC5, %edi
	call	puts
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	b5, .-b5
	.globl	b6
	.section	.dtors,"aw",@progbits
	.align 8
	.type	b6, @object
	.size	b6, 8
b6:
	.quad	b5
	.text
	.globl	regiester_atexit
	.type	regiester_atexit, @function
regiester_atexit:
.LFB8:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$f1, %edi
	call	atexit
	movl	$f2, %edi
	call	atexit
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	regiester_atexit, .-regiester_atexit
	.section	.rodata
.LC6:
	.string	"exiting"
	.text
	.globl	main
	.type	main, @function
main:
.LFB9:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$0, %eax
	call	regiester_atexit
	movl	$.LC6, %edi
	call	puts
	movl	$0, %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609"
	.section	.note.GNU-stack,"",@progbits
