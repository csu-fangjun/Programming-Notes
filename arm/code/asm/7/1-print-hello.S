
/*
// we use .S instead of .s as file extension
// so that the preprocessor is invoked by the compiler.
// Therefore, we can use // and and c style comment;
// we can even define macros as in c/c++.

// x29: frame pointer
// x30: link register
// since we invoke `printf` inside main, we
// have to save x29 and x30 manually
//
// Note that sp has to be 16-byte aligned, so we have
// to push a pair of 16-byte registers into the stack;
//
// it is not possible to push a single 16-byte register
// to the stack

*/

  .section .rodata

hello:
  .asciz "1234" // z in asciz means it is implicitly 0 terminated

// the use of `=` enables us to use `mov`
len_hello = . - hello


new_line:
  .ascii "5678" // noe that ascii is not zero terminated!
	.string "!\n" // string is equivalent to asciz

// we have to use adr + ldrb to get the value of len_new_line
len_new_line:
	.byte . - new_line

len_msg1:
	.string "len of hello: %d\n"

len_msg2:
	.string "len of new_line: %d\n"

sys_msg:
	.string "hello write\n"
sys_msg_len = . - sys_msg

  .section .text
  .global main
  .type main, %function


main:
  stp x29, x30, [sp, -16]!    // we have to save x29(frame pointer) and x30 (return address)
                              // since we call printf below.

  adr x0, hello // load the address of hello into x0
  bl printf   // we can call the c function printf without pre-declaration.

  adr x0, new_line  // load the address of new_line into x0
  bl printf

  adr x0, len_msg1
  mov w1, len_hello
  bl printf

  adr x0, len_new_line
  ldrb w1, [x0]
  adr x0, len_msg2
  bl printf


	// now use system call: ssize_t write(int fd, const void* buf, size_t count)
  // man syscall
  // says that x8 contains the syscall number
	mov w0, 2	// stdout
	adr x1, sys_msg
	mov x2, sys_msg_len
	mov x8, 64  // https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/unistd.h#L207
              // the call number is 64 or write
              // man sys
              // will show that x8 should contain the call number
  svc 0


  mov x0, 0		// exit code, e.g., exit(0)
  ldp x29, x30, [sp], 16
  ret
  .size main, . - main


