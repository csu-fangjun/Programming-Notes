	.cpu generic+fp+simd
	.file	"inline_asm.cc"
	.section	.rodata
	.align	3
.LC0:
	.string	"inline_asm.cc"
	.align	3
.LC1:
	.string	"a == b"
	.align	3
.LC2:
	.string	"(a + 1) == b"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_19test_moveEv, %function
_ZN12_GLOBAL__N_19test_moveEv:
.LFB0:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 10
	str	w0, [x29, 44]
	ldr	w0, [x29, 44]
#APP
// 22 "inline_asm.cc" 1
	mov w0, w0
// 0 "" 2
#NO_APP
	str	w0, [x29, 40]
	ldr	w1, [x29, 44]
	ldr	w0, [x29, 40]
	cmp	w1, w0
	beq	.L2
	adrp	x0, _ZZN12_GLOBAL__N_19test_moveEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_19test_moveEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC1
	add	x0, x0, :lo12:.LC1
	mov	x3, x2
	mov	w2, 23
	bl	__assert_fail
.L2:
	mov	x0, 48879
	movk	x0, 0xdead, lsl 16
	movk	x0, 0x5678, lsl 32
	movk	x0, 0x1234, lsl 48
	str	x0, [x29, 32]
	ldr	x0, [x29, 32]
#APP
// 60 "inline_asm.cc" 1
	mov x0, x0     
add x0, x0, 1  

// 0 "" 2
#NO_APP
	str	x0, [x29, 24]
	ldr	x0, [x29, 32]
	add	x1, x0, 1
	ldr	x0, [x29, 24]
	cmp	x1, x0
	beq	.L1
	adrp	x0, _ZZN12_GLOBAL__N_19test_moveEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_19test_moveEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	mov	x3, x2
	mov	w2, 61
	bl	__assert_fail
.L1:
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE0:
	.size	_ZN12_GLOBAL__N_19test_moveEv, .-_ZN12_GLOBAL__N_19test_moveEv
	.section	.rodata
	.align	3
.LC3:
	.string	"a == 2"
	.align	3
.LC4:
	.string	"b == 12"
	.align	3
.LC5:
	.string	"c == 103"
	.align	3
.LC6:
	.string	"sum == 30"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_18test_addEv, %function
_ZN12_GLOBAL__N_18test_addEv:
.LFB1:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 1
	str	w0, [x29, 44]
	mov	w0, 2
	str	w0, [x29, 40]
	mov	w0, 3
	str	w0, [x29, 36]
	ldr	w2, [x29, 44]
	ldr	w1, [x29, 40]
	ldr	w0, [x29, 36]
	mov	w3, w2
	mov	w2, w1
	mov	w1, w0
#APP
// 119 "inline_asm.cc" 1
	add w3, w3, 1    
add w2, w2, 10   
add w1, w1, 100  

// 0 "" 2
#NO_APP
	str	w3, [x29, 44]
	str	w2, [x29, 40]
	str	w1, [x29, 36]
	ldr	w0, [x29, 44]
	cmp	w0, 2
	beq	.L5
	adrp	x0, _ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC3
	add	x0, x0, :lo12:.LC3
	mov	x3, x2
	mov	w2, 120
	bl	__assert_fail
.L5:
	ldr	w0, [x29, 40]
	cmp	w0, 12
	beq	.L6
	adrp	x0, _ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC4
	add	x0, x0, :lo12:.LC4
	mov	x3, x2
	mov	w2, 121
	bl	__assert_fail
.L6:
	ldr	w0, [x29, 36]
	cmp	w0, 103
	beq	.L7
	adrp	x0, _ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC5
	add	x0, x0, :lo12:.LC5
	mov	x3, x2
	mov	w2, 122
	bl	__assert_fail
.L7:
	mov	w0, 10
	str	w0, [x29, 32]
	mov	w0, 20
	str	w0, [x29, 28]
	mov	w0, -1
	str	w0, [x29, 24]
	ldr	w0, [x29, 32]
	ldr	w1, [x29, 28]
#APP
// 187 "inline_asm.cc" 1
	add x0, x0, x1   

// 0 "" 2
#NO_APP
	str	w0, [x29, 24]
	ldr	w0, [x29, 24]
	cmp	w0, 30
	beq	.L4
	adrp	x0, _ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC6
	add	x0, x0, :lo12:.LC6
	mov	x3, x2
	mov	w2, 188
	bl	__assert_fail
.L4:
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE1:
	.size	_ZN12_GLOBAL__N_18test_addEv, .-_ZN12_GLOBAL__N_18test_addEv
	.section	.rodata
	.align	3
.LC7:
	.string	"b[0] == 10"
	.align	3
.LC8:
	.string	"b[1] == 20"
	.align	3
.LC9:
	.string	"c[0] == 30"
	.align	3
.LC10:
	.string	"c[1] == 40"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_114test_load_pairEv, %function
_ZN12_GLOBAL__N_114test_load_pairEv:
.LFB2:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 10
	str	w0, [x29, 32]
	mov	w0, 20
	str	w0, [x29, 36]
	mov	w0, 30
	str	w0, [x29, 40]
	mov	w0, 40
	str	w0, [x29, 44]
	str	xzr, [x29, 24]
	str	xzr, [x29, 16]
	add	x2, x29, 32
	add	x3, x29, 24
	add	x4, x29, 16
#APP
// 233 "inline_asm.cc" 1
	ldp x0, x1, [x2]   
str x0, [x3]       
str x1, [x4]       

// 0 "" 2
#NO_APP
	ldr	w0, [x29, 24]
	cmp	w0, 10
	beq	.L10
	adrp	x0, _ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC7
	add	x0, x0, :lo12:.LC7
	mov	x3, x2
	mov	w2, 235
	bl	__assert_fail
.L10:
	ldr	w0, [x29, 28]
	cmp	w0, 20
	beq	.L11
	adrp	x0, _ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC8
	add	x0, x0, :lo12:.LC8
	mov	x3, x2
	mov	w2, 236
	bl	__assert_fail
.L11:
	ldr	w0, [x29, 16]
	cmp	w0, 30
	beq	.L12
	adrp	x0, _ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC9
	add	x0, x0, :lo12:.LC9
	mov	x3, x2
	mov	w2, 238
	bl	__assert_fail
.L12:
	ldr	w0, [x29, 20]
	cmp	w0, 40
	beq	.L9
	adrp	x0, _ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC10
	add	x0, x0, :lo12:.LC10
	mov	x3, x2
	mov	w2, 239
	bl	__assert_fail
.L9:
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE2:
	.size	_ZN12_GLOBAL__N_114test_load_pairEv, .-_ZN12_GLOBAL__N_114test_load_pairEv
	.section	.rodata
	.align	3
.LC11:
	.string	"sum[0] == 18"
	.align	3
.LC12:
	.string	"sum[1] == 28"
	.align	3
.LC13:
	.string	"sum[2] == 38"
	.align	3
.LC14:
	.string	"sum[3] == 58"
	.align	3
.LC15:
	.string	"c[0] == 11"
	.align	3
.LC16:
	.string	"c[1] == 12"
	.align	3
.LC17:
	.string	"c[2] == 13"
	.align	3
.LC18:
	.string	"c[3] == 15"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_18test_dupEv, %function
_ZN12_GLOBAL__N_18test_dupEv:
.LFB3:
	.cfi_startproc
	stp	x29, x30, [sp, -80]!
	.cfi_def_cfa_offset 80
	.cfi_offset 29, -80
	.cfi_offset 30, -72
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 10
	str	w0, [x29, 56]
	mov	w0, 20
	str	w0, [x29, 60]
	mov	w0, 30
	str	w0, [x29, 64]
	mov	w0, 50
	str	w0, [x29, 68]
	mov	w0, 8
	str	w0, [x29, 48]
	add	x0, x29, 56
	add	x1, x29, 48
	add	x2, x29, 32
#APP
// 305 "inline_asm.cc" 1
	ld1 {v0.4s}, [x0]       
ldr s1, [x1]            
dup v2.4s, v1.s[0]        
add v1.4s, v0.4s, v2.4s   
st1 {v1.4s}, [x2]     

// 0 "" 2
#NO_APP
	ldr	w0, [x29, 32]
	cmp	w0, 18
	beq	.L15
	adrp	x0, _ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC11
	add	x0, x0, :lo12:.LC11
	mov	x3, x2
	mov	w2, 306
	bl	__assert_fail
.L15:
	ldr	w0, [x29, 36]
	cmp	w0, 28
	beq	.L16
	adrp	x0, _ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC12
	add	x0, x0, :lo12:.LC12
	mov	x3, x2
	mov	w2, 307
	bl	__assert_fail
.L16:
	ldr	w0, [x29, 40]
	cmp	w0, 38
	beq	.L17
	adrp	x0, _ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC13
	add	x0, x0, :lo12:.LC13
	mov	x3, x2
	mov	w2, 308
	bl	__assert_fail
.L17:
	ldr	w0, [x29, 44]
	cmp	w0, 58
	beq	.L18
	adrp	x0, _ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC14
	add	x0, x0, :lo12:.LC14
	mov	x3, x2
	mov	w2, 309
	bl	__assert_fail
.L18:
	mov	w0, 1
	strh	w0, [x29, 24]
	mov	w0, 2
	strh	w0, [x29, 26]
	mov	w0, 3
	strh	w0, [x29, 28]
	mov	w0, 5
	strh	w0, [x29, 30]
	mov	w0, 10
	strb	w0, [x29, 79]
	add	x0, x29, 24
	ldrb	w1, [x29, 79]
	add	x2, x29, 16
#APP
// 370 "inline_asm.cc" 1
	ld1 {v0.4h}, [x0]      
dup v2.4h, w1         
add v1.4h, v0.4h, v2.4h  
st1 {v1.4h}, [x2]      

// 0 "" 2
#NO_APP
	ldrsh	w0, [x29, 16]
	cmp	w0, 11
	beq	.L19
	adrp	x0, _ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC15
	add	x0, x0, :lo12:.LC15
	mov	x3, x2
	mov	w2, 371
	bl	__assert_fail
.L19:
	ldrsh	w0, [x29, 18]
	cmp	w0, 12
	beq	.L20
	adrp	x0, _ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC16
	add	x0, x0, :lo12:.LC16
	mov	x3, x2
	mov	w2, 372
	bl	__assert_fail
.L20:
	ldrsh	w0, [x29, 20]
	cmp	w0, 13
	beq	.L21
	adrp	x0, _ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC17
	add	x0, x0, :lo12:.LC17
	mov	x3, x2
	mov	w2, 373
	bl	__assert_fail
.L21:
	ldrsh	w0, [x29, 22]
	cmp	w0, 15
	beq	.L14
	adrp	x0, _ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC18
	add	x0, x0, :lo12:.LC18
	mov	x3, x2
	mov	w2, 374
	bl	__assert_fail
.L14:
	ldp	x29, x30, [sp], 80
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE3:
	.size	_ZN12_GLOBAL__N_18test_dupEv, .-_ZN12_GLOBAL__N_18test_dupEv
	.section	.rodata
	.align	3
.LC19:
	.string	"a == 0x78563412"
	.align	3
.LC20:
	.string	"a == 0x34120000"
	.align	3
.LC21:
	.string	"b == 0x3412"
	.align	3
.LC22:
	.string	"b == 0x12"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_119test_swap_endianessEv, %function
_ZN12_GLOBAL__N_119test_swap_endianessEv:
.LFB4:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 22136
	movk	w0, 0x1234, lsl 16
	str	w0, [x29, 28]
	ldr	w0, [x29, 28]
#APP
// 412 "inline_asm.cc" 1
	rev w0, w0 

// 0 "" 2
#NO_APP
	str	w0, [x29, 28]
	ldr	w1, [x29, 28]
	mov	w0, 13330
	movk	w0, 0x7856, lsl 16
	cmp	w1, w0
	beq	.L24
	adrp	x0, _ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC19
	add	x0, x0, :lo12:.LC19
	mov	x3, x2
	mov	w2, 413
	bl	__assert_fail
.L24:
	mov	w0, 4660
	str	w0, [x29, 28]
	ldr	w0, [x29, 28]
#APP
// 416 "inline_asm.cc" 1
	rev w0, w0 

// 0 "" 2
#NO_APP
	str	w0, [x29, 28]
	ldr	w1, [x29, 28]
	mov	w0, 873594880
	cmp	w1, w0
	beq	.L25
	adrp	x0, _ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC20
	add	x0, x0, :lo12:.LC20
	mov	x3, x2
	mov	w2, 417
	bl	__assert_fail
.L25:
	mov	w0, 4660
	strh	w0, [x29, 26]
	ldrh	w0, [x29, 26]
#APP
// 423 "inline_asm.cc" 1
	rev16 w0, w0 

// 0 "" 2
#NO_APP
	strh	w0, [x29, 26]
	ldrsh	w1, [x29, 26]
	mov	w0, 13330
	cmp	w1, w0
	beq	.L26
	adrp	x0, _ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC21
	add	x0, x0, :lo12:.LC21
	mov	x3, x2
	mov	w2, 424
	bl	__assert_fail
.L26:
	mov	w0, 4608
	strh	w0, [x29, 26]
	ldrh	w0, [x29, 26]
#APP
// 430 "inline_asm.cc" 1
	rev16 w0, w0 

// 0 "" 2
#NO_APP
	strh	w0, [x29, 26]
	ldrsh	w0, [x29, 26]
	cmp	w0, 18
	beq	.L23
	adrp	x0, _ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC22
	add	x0, x0, :lo12:.LC22
	mov	x3, x2
	mov	w2, 431
	bl	__assert_fail
.L23:
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE4:
	.size	_ZN12_GLOBAL__N_119test_swap_endianessEv, .-_ZN12_GLOBAL__N_119test_swap_endianessEv
	.section	.rodata
	.align	3
.LC23:
	.string	"b == (1 | (1 << 2) | (1 << 8))"
	.align	3
.LC24:
	.string	"b == 8"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_18test_bicEv, %function
_ZN12_GLOBAL__N_18test_bicEv:
.LFB5:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 277
	str	w0, [x29, 28]
	mov	w0, 16
	str	w0, [x29, 24]
	ldr	w0, [x29, 28]
	ldr	w1, [x29, 24]
#APP
// 444 "inline_asm.cc" 1
	bic w0, w0, w1 

// 0 "" 2
#NO_APP
	str	w0, [x29, 20]
	ldr	w0, [x29, 20]
	cmp	w0, 261
	beq	.L29
	adrp	x0, _ZZN12_GLOBAL__N_18test_bicEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_bicEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC23
	add	x0, x0, :lo12:.LC23
	mov	x3, x2
	mov	w2, 445
	bl	__assert_fail
.L29:
	mov	w0, 10
	str	w0, [x29, 28]
	mov	w0, 3
	str	w0, [x29, 24]
	ldr	w0, [x29, 28]
	ldr	w1, [x29, 24]
#APP
// 452 "inline_asm.cc" 1
	bic w0, w0, w1 

// 0 "" 2
#NO_APP
	str	w0, [x29, 20]
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	.L28
	adrp	x0, _ZZN12_GLOBAL__N_18test_bicEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_bicEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC24
	add	x0, x0, :lo12:.LC24
	mov	x3, x2
	mov	w2, 453
	bl	__assert_fail
.L28:
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE5:
	.size	_ZN12_GLOBAL__N_18test_bicEv, .-_ZN12_GLOBAL__N_18test_bicEv
	.section	.rodata
	.align	3
.LC25:
	.string	"c == (0x01 | 0x03)"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_18test_orrEv, %function
_ZN12_GLOBAL__N_18test_orrEv:
.LFB6:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 1
	str	w0, [x29, 28]
	mov	w0, 3
	str	w0, [x29, 24]
	ldr	w0, [x29, 28]
	ldr	w1, [x29, 24]
#APP
// 464 "inline_asm.cc" 1
	orr w0, w0, w1   

// 0 "" 2
#NO_APP
	str	w0, [x29, 20]
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	.L31
	adrp	x0, _ZZN12_GLOBAL__N_18test_orrEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_orrEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC25
	add	x0, x0, :lo12:.LC25
	mov	x3, x2
	mov	w2, 465
	bl	__assert_fail
.L31:
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE6:
	.size	_ZN12_GLOBAL__N_18test_orrEv, .-_ZN12_GLOBAL__N_18test_orrEv
	.section	.rodata
	.align	3
.LC26:
	.string	"c == (0x01 ^ 0x03)"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_18test_eorEv, %function
_ZN12_GLOBAL__N_18test_eorEv:
.LFB7:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 1
	str	w0, [x29, 28]
	mov	w0, 3
	str	w0, [x29, 24]
	ldr	w0, [x29, 28]
	ldr	w1, [x29, 24]
#APP
// 476 "inline_asm.cc" 1
	eor w0, w0, w1   

// 0 "" 2
#NO_APP
	str	w0, [x29, 20]
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	.L33
	adrp	x0, _ZZN12_GLOBAL__N_18test_eorEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_eorEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC26
	add	x0, x0, :lo12:.LC26
	mov	x3, x2
	mov	w2, 477
	bl	__assert_fail
.L33:
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE7:
	.size	_ZN12_GLOBAL__N_18test_eorEv, .-_ZN12_GLOBAL__N_18test_eorEv
	.section	.rodata
	.align	3
.LC27:
	.string	"c == (0x01 | ~0x03)"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_18test_ornEv, %function
_ZN12_GLOBAL__N_18test_ornEv:
.LFB8:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 1
	str	w0, [x29, 28]
	mov	w0, 3
	str	w0, [x29, 24]
	ldr	w0, [x29, 28]
	ldr	w1, [x29, 24]
#APP
// 488 "inline_asm.cc" 1
	orn w0, w0, w1   

// 0 "" 2
#NO_APP
	str	w0, [x29, 20]
	ldr	w0, [x29, 20]
	cmn	w0, #3
	beq	.L35
	adrp	x0, _ZZN12_GLOBAL__N_18test_ornEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_ornEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC27
	add	x0, x0, :lo12:.LC27
	mov	x3, x2
	mov	w2, 489
	bl	__assert_fail
.L35:
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE8:
	.size	_ZN12_GLOBAL__N_18test_ornEv, .-_ZN12_GLOBAL__N_18test_ornEv
	.section	.rodata
	.align	3
.LC28:
	.string	"f == d"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_19test_ccmpEv, %function
_ZN12_GLOBAL__N_19test_ccmpEv:
.LFB9:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 2
	str	w0, [x29, 40]
	mov	w0, 1
	str	w0, [x29, 36]
	mov	w0, 1
	str	w0, [x29, 32]
	ldr	w1, [x29, 40]
	ldr	w0, [x29, 36]
	cmp	w1, w0
	ble	.L38
	ldr	w1, [x29, 36]
	ldr	w0, [x29, 32]
	cmp	w1, w0
	bne	.L38
	mov	w0, 10
	str	w0, [x29, 44]
	b	.L39
.L38:
	mov	w0, 100
	str	w0, [x29, 44]
.L39:
	mov	w0, -1
	str	w0, [x29, 28]
	ldr	w0, [x29, 40]
	ldr	w1, [x29, 36]
	ldr	w2, [x29, 32]
#APP
// 528 "inline_asm.cc" 1
	cmp w0, w1         
ccmp w1, w2, 0, gt 
b.eq 1f                  
mov w0, 100           
b 2f                     
1:                       
mov w0, 10            
2:                       

// 0 "" 2
#NO_APP
	str	w0, [x29, 28]
	ldr	w1, [x29, 28]
	ldr	w0, [x29, 44]
	cmp	w1, w0
	beq	.L37
	adrp	x0, _ZZN12_GLOBAL__N_19test_ccmpEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_19test_ccmpEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC28
	add	x0, x0, :lo12:.LC28
	mov	x3, x2
	mov	w2, 529
	bl	__assert_fail
.L37:
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE9:
	.size	_ZN12_GLOBAL__N_19test_ccmpEv, .-_ZN12_GLOBAL__N_19test_ccmpEv
	.section	.rodata
	.align	3
.LC29:
	.string	"r == 31"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_18test_clzEv, %function
_ZN12_GLOBAL__N_18test_clzEv:
.LFB10:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 1
	str	w0, [x29, 28]
	ldr	w0, [x29, 28]
#APP
// 537 "inline_asm.cc" 1
	clz w0, w0

// 0 "" 2
#NO_APP
	str	w0, [x29, 24]
	ldr	w0, [x29, 24]
	cmp	w0, 31
	beq	.L41
	adrp	x0, _ZZN12_GLOBAL__N_18test_clzEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_18test_clzEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC29
	add	x0, x0, :lo12:.LC29
	mov	x3, x2
	mov	w2, 539
	bl	__assert_fail
.L41:
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE10:
	.size	_ZN12_GLOBAL__N_18test_clzEv, .-_ZN12_GLOBAL__N_18test_clzEv
	.section	.rodata
	.align	3
.LC30:
	.string	"r == f"
	.text
	.align	2
	.type	_ZN12_GLOBAL__N_19test_cselEv, %function
_ZN12_GLOBAL__N_19test_cselEv:
.LFB11:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	mov	w0, 10
	str	w0, [x29, 28]
	mov	w0, 20
	str	w0, [x29, 24]
	ldr	w0, [x29, 28]
	cmp	w0, 9
	ble	.L44
	ldr	w0, [x29, 28]
	b	.L45
.L44:
	ldr	w0, [x29, 24]
.L45:
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	ldr	w1, [x29, 24]
#APP
// 557 "inline_asm.cc" 1
	cmp w0, 10                
csel w0, w0, w1, ge 

// 0 "" 2
#NO_APP
	str	w0, [x29, 16]
	ldr	w1, [x29, 20]
	ldr	w0, [x29, 16]
	cmp	w1, w0
	beq	.L43
	adrp	x0, _ZZN12_GLOBAL__N_19test_cselEvE19__PRETTY_FUNCTION__
	add	x2, x0, :lo12:_ZZN12_GLOBAL__N_19test_cselEvE19__PRETTY_FUNCTION__
	adrp	x0, .LC0
	add	x1, x0, :lo12:.LC0
	adrp	x0, .LC30
	add	x0, x0, :lo12:.LC30
	mov	x3, x2
	mov	w2, 558
	bl	__assert_fail
.L43:
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE11:
	.size	_ZN12_GLOBAL__N_19test_cselEv, .-_ZN12_GLOBAL__N_19test_cselEv
	.align	2
	.global	_Z15test_inline_asmv
	.type	_Z15test_inline_asmv, %function
_Z15test_inline_asmv:
.LFB12:
	.cfi_startproc
	stp	x29, x30, [sp, -16]!
	.cfi_def_cfa_offset 16
	.cfi_offset 29, -16
	.cfi_offset 30, -8
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	bl	_ZN12_GLOBAL__N_19test_moveEv
	bl	_ZN12_GLOBAL__N_18test_addEv
	bl	_ZN12_GLOBAL__N_114test_load_pairEv
	bl	_ZN12_GLOBAL__N_18test_dupEv
	bl	_ZN12_GLOBAL__N_119test_swap_endianessEv
	bl	_ZN12_GLOBAL__N_18test_bicEv
	bl	_ZN12_GLOBAL__N_18test_orrEv
	bl	_ZN12_GLOBAL__N_18test_eorEv
	bl	_ZN12_GLOBAL__N_18test_ornEv
	bl	_ZN12_GLOBAL__N_19test_ccmpEv
	bl	_ZN12_GLOBAL__N_18test_clzEv
	bl	_ZN12_GLOBAL__N_19test_cselEv
	ldp	x29, x30, [sp], 16
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE12:
	.size	_Z15test_inline_asmv, .-_Z15test_inline_asmv
	.section	.rodata
	.align	3
	.type	_ZZN12_GLOBAL__N_19test_moveEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_19test_moveEvE19__PRETTY_FUNCTION__, 30
_ZZN12_GLOBAL__N_19test_moveEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_move()"
	.align	3
	.type	_ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__, 29
_ZZN12_GLOBAL__N_18test_addEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_add()"
	.align	3
	.type	_ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__, 35
_ZZN12_GLOBAL__N_114test_load_pairEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_load_pair()"
	.align	3
	.type	_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__, 29
_ZZN12_GLOBAL__N_18test_dupEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_dup()"
	.align	3
	.type	_ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__, 40
_ZZN12_GLOBAL__N_119test_swap_endianessEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_swap_endianess()"
	.align	3
	.type	_ZZN12_GLOBAL__N_18test_bicEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_18test_bicEvE19__PRETTY_FUNCTION__, 29
_ZZN12_GLOBAL__N_18test_bicEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_bic()"
	.align	3
	.type	_ZZN12_GLOBAL__N_18test_orrEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_18test_orrEvE19__PRETTY_FUNCTION__, 29
_ZZN12_GLOBAL__N_18test_orrEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_orr()"
	.align	3
	.type	_ZZN12_GLOBAL__N_18test_eorEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_18test_eorEvE19__PRETTY_FUNCTION__, 29
_ZZN12_GLOBAL__N_18test_eorEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_eor()"
	.align	3
	.type	_ZZN12_GLOBAL__N_18test_ornEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_18test_ornEvE19__PRETTY_FUNCTION__, 29
_ZZN12_GLOBAL__N_18test_ornEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_orn()"
	.align	3
	.type	_ZZN12_GLOBAL__N_19test_ccmpEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_19test_ccmpEvE19__PRETTY_FUNCTION__, 30
_ZZN12_GLOBAL__N_19test_ccmpEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_ccmp()"
	.align	3
	.type	_ZZN12_GLOBAL__N_18test_clzEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_18test_clzEvE19__PRETTY_FUNCTION__, 29
_ZZN12_GLOBAL__N_18test_clzEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_clz()"
	.align	3
	.type	_ZZN12_GLOBAL__N_19test_cselEvE19__PRETTY_FUNCTION__, %object
	.size	_ZZN12_GLOBAL__N_19test_cselEvE19__PRETTY_FUNCTION__, 30
_ZZN12_GLOBAL__N_19test_cselEvE19__PRETTY_FUNCTION__:
	.string	"void {anonymous}::test_csel()"
	.ident	"GCC: (Linaro GCC 4.9-2016.02) 4.9.4 20151028 (prerelease)"
	.section	.note.GNU-stack,"",%progbits
