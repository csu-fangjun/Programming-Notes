	.file	"init.cc"
	.section	.text._ZN3FooC2Ev,"axG",@progbits,Foo::Foo(),comdat
	.align 2
	.weak	Foo::Foo()
	.type	Foo::Foo(), @function
Foo::Foo():
.LFB1:
	pushq	%rbp
.LCFI0:
	movq	%rsp, %rbp
.LCFI1:
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	$12345, (%rax)
	nop
	popq	%rbp
.LCFI2:
	ret
.LFE1:
	.size	Foo::Foo(), .-Foo::Foo()
	.weak	Foo::Foo()
	.set	Foo::Foo(),Foo::Foo()
	.section	.text._ZN3FooD2Ev,"axG",@progbits,Foo::~Foo(),comdat
	.align 2
	.weak	Foo::~Foo()
	.type	Foo::~Foo(), @function
Foo::~Foo():
.LFB4:
	pushq	%rbp
.LCFI3:
	movq	%rsp, %rbp
.LCFI4:
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	$54321, (%rax)
	nop
	popq	%rbp
.LCFI5:
	ret
.LFE4:
	.size	Foo::~Foo(), .-Foo::~Foo()
	.weak	Foo::~Foo()
	.set	Foo::~Foo(),Foo::~Foo()
	.section	.text._ZNK3Foo3GetEv,"axG",@progbits,Foo::Get() const,comdat
	.align 2
	.weak	Foo::Get() const
	.type	Foo::Get() const, @function
Foo::Get() const:
.LFB6:
	pushq	%rbp
.LCFI6:
	movq	%rsp, %rbp
.LCFI7:
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	(%rax), %eax
	popq	%rbp
.LCFI8:
	ret
.LFE6:
	.size	Foo::Get() const, .-Foo::Get() const
	.globl	f
	.bss
	.align 4
	.type	f, @object
	.size	f, 4
f:
	.zero	4
	.text
	.globl	test()
	.type	test(), @function
test():
.LFB7:
	pushq	%rbp
.LCFI9:
	movq	%rsp, %rbp
.LCFI10:
	movl	$f, %edi
	call	Foo::Get() const
	popq	%rbp
.LCFI11:
	ret
.LFE7:
	.size	test(), .-test()
	.type	__static_initialization_and_destruction_0(int, int), @function
__static_initialization_and_destruction_0(int, int):
.LFB8:
	pushq	%rbp
.LCFI12:
	movq	%rsp, %rbp
.LCFI13:
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	cmpl	$1, -4(%rbp)
	jne	.L9
	cmpl	$65535, -8(%rbp)
	jne	.L9
	movl	$f, %edi
	call	Foo::Foo()
	movl	$__dso_handle, %edx
	movl	$f, %esi
	movl	Foo::~Foo(), %edi
	call	__cxa_atexit
.L9:
	nop
	leave
.LCFI14:
	ret
.LFE8:
	.size	__static_initialization_and_destruction_0(int, int), .-__static_initialization_and_destruction_0(int, int)
	.type	_GLOBAL__sub_I_f, @function
_GLOBAL__sub_I_f:
.LFB9:
	pushq	%rbp
.LCFI15:
	movq	%rsp, %rbp
.LCFI16:
	movl	$65535, %esi
	movl	$1, %edi
	call	__static_initialization_and_destruction_0(int, int)
	popq	%rbp
.LCFI17:
	ret
.LFE9:
	.size	_GLOBAL__sub_I_f, .-_GLOBAL__sub_I_f
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_f
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609"
	.section	.note.GNU-stack,"",@progbits
