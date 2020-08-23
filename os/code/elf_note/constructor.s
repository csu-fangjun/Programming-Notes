	.file	"constructor.cc"
	.text
	.globl	_Z8test_conv
	.type	_Z8test_conv, @function
_Z8test_conv:
.LFB0:
	pushq	%rbp
.LCFI0:
	movq	%rsp, %rbp
.LCFI1:
	popq	%rbp
.LCFI2:
	ret
.LFE0:
	.size	_Z8test_conv, .-_Z8test_conv
	.section	.init_array,"aw"
	.align 8
	.quad	_Z8test_conv
	.text
	.globl	_Z9test_con2v
	.type	_Z9test_con2v, @function
_Z9test_con2v:
.LFB1:
	pushq	%rbp
.LCFI3:
	movq	%rsp, %rbp
.LCFI4:
	popq	%rbp
.LCFI5:
	ret
.LFE1:
	.size	_Z9test_con2v, .-_Z9test_con2v
	.section	.init_array
	.align 8
	.quad	_Z9test_con2v
	.text
	.globl	_Z8test_desv
	.type	_Z8test_desv, @function
_Z8test_desv:
.LFB2:
	pushq	%rbp
.LCFI6:
	movq	%rsp, %rbp
.LCFI7:
	popq	%rbp
.LCFI8:
	ret
.LFE2:
	.size	_Z8test_desv, .-_Z8test_desv
	.section	.fini_array,"aw"
	.align 8
	.quad	_Z8test_desv
	.text
	.globl	_Z9test_des2v
	.type	_Z9test_des2v, @function
_Z9test_des2v:
.LFB3:
	pushq	%rbp
.LCFI9:
	movq	%rsp, %rbp
.LCFI10:
	popq	%rbp
.LCFI11:
	ret
.LFE3:
	.size	_Z9test_des2v, .-_Z9test_des2v
	.section	.fini_array
	.align 8
	.quad	_Z9test_des2v
	.ident	"GCC: (Ubuntu 4.8.4-2ubuntu1~14.04.4) 4.8.4"
	.section	.note.GNU-stack,"",@progbits
